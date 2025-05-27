import os
import contextlib
import typing
from typing import Optional
import warnings
import logging
import time

import numpy as np
import h5py
from sparseconverter import CUDA, NUMPY, ArrayBackend

from libertem.common.math import prod, flat_nonzero
from libertem.common import Slice, Shape
from libertem.common.buffers import zeros_aligned
from libertem.io.corrections import CorrectionSet
from libertem.common.messageconverter import MessageConverter
from .base import (
    DataSet, Partition, DataTile, DataSetException, DataSetMeta,
    TilingScheme,
)


# alias for mocking:
current_time = time.time

logger = logging.getLogger(__name__)


class HDF5DatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/HDF5DatasetParams.schema.json",
        "title": "HDF5DatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "HDF5"},
            "path": {"type": "string"},
            "ds_path": {"type": "string"},
            "nav_shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "sig_shape": {
                "type": "array",
                "items": {"type": "number", "minimum": 1},
                "minItems": 2,
                "maxItems": 2
            },
            "sync_offset": {"type": "number"},
        },
        "required": ["type", "path", "ds_path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path", "ds_path"]
        }
        if "nav_shape" in raw_data:
            data["nav_shape"] = tuple(raw_data["nav_shape"])
        if "sig_shape" in raw_data:
            data["sig_shape"] = tuple(raw_data["sig_shape"])
        if "sync_offset" in raw_data:
            data["sync_offset"] = raw_data["sync_offset"]
        return data


def _ensure_2d_nav(nav_shape: tuple[int, ...]) -> tuple[int, int]:
    # For any iterable shape, reduce or pad it to a 2-tuple
    # with the same prod(shape). Reduction from left to right
    # (final dimension preserved). Special case for empty
    # nav_shape which is converted to (1, 1)
    nav_shape = tuple(nav_shape)
    if len(nav_shape) == 1:
        nav_shape = (1,) + nav_shape
    elif len(nav_shape) >= 2:
        nav_shape = (prod(nav_shape[:-1]),) + nav_shape[-1:]
    elif len(nav_shape) == 0:
        return (1, 1)
    else:
        raise ValueError(f'Incompatible nav_shape {nav_shape}')
    return nav_shape


class HDF5ArrayDescriptor(typing.NamedTuple):
    name: str
    shape: tuple[int, ...]
    dtype: np.dtype
    compression: Optional[str]
    chunks: tuple[int, ...]


def _get_datasets(path):
    datasets: list[HDF5ArrayDescriptor] = []

    try:
        timeout = int(os.environ.get('LIBERTEM_IO_HDF5_TIMEOUT_DEBUG', 3))
    except ValueError:
        timeout = 3

    t0 = current_time()

    def _make_list(name, obj):
        if current_time() - t0 > timeout:
            raise TimeoutError
        if hasattr(obj, 'size') and hasattr(obj, 'shape'):
            if obj.ndim < 3:
                # Can't process this dataset, skip
                return
            datasets.append(
                HDF5ArrayDescriptor(name, obj.shape, obj.dtype, obj.compression, obj.chunks)
            )

    with h5py.File(path, 'r') as f:
        f.visititems(_make_list)
    return datasets


def _have_contig_chunks(chunks, ds_shape):
    """
    Returns `True` if the `chunks` are contiguous in the navigation axes.

    Examples
    --------

    >>> ds_shape = Shape((64, 64, 128, 128), sig_dims=2)
    >>> _have_contig_chunks((1, 4, 32, 32), ds_shape)
    True
    >>> _have_contig_chunks((2, 4, 32, 32), ds_shape)
    False
    >>> _have_contig_chunks((2, 64, 32, 32), ds_shape)
    True
    >>> _have_contig_chunks((64, 1, 32, 32), ds_shape)
    False
    >>> ds_shape_5d = Shape((16, 64, 64, 128, 128), sig_dims=2)
    >>> _have_contig_chunks((1, 1, 2, 32, 32), ds_shape_5d)
    True
    >>> _have_contig_chunks((1, 2, 1, 32, 32), ds_shape_5d)
    False
    >>> _have_contig_chunks((2, 1, 1, 32, 32), ds_shape_5d)
    False
    """
    # In other terms:
    # There exists an index `i` such that `prod(chunks[:i]) == 1` and
    # chunks[i+1:] == ds_shape[i+1:], limited to the nav part of chunks and ds_shape
    #
    nav_shape = tuple(ds_shape.nav)
    nav_dims = len(nav_shape)
    chunks_nav = chunks[:nav_dims]

    for i in range(nav_dims):
        left = chunks_nav[:i]
        left_prod = prod(left)
        if left_prod == 1 and chunks_nav[i + 1:] == nav_shape[i + 1:]:
            return True
    return False


def _partition_shape_for_chunking(chunks, ds_shape):
    """
    Get the minimum partition shape for that allows us to prevent read amplification
    with chunked HDF5 files.

    Examples
    --------

    >>> ds_shape = Shape((64, 64, 128, 128), sig_dims=2)
    >>> _partition_shape_for_chunking((1, 4, 32, 32), ds_shape)
    (1, 4, 128, 128)
    >>> _partition_shape_for_chunking((2, 4, 32, 32), ds_shape)
    (2, 64, 128, 128)
    >>> _partition_shape_for_chunking((2, 64, 32, 32), ds_shape)
    (2, 64, 128, 128)
    >>> _partition_shape_for_chunking((64, 1, 32, 32), ds_shape)
    (64, 64, 128, 128)
    >>> ds_shape_5d = Shape((16, 64, 64, 128, 128), sig_dims=2)
    >>> _partition_shape_for_chunking((1, 1, 2, 32, 32), ds_shape_5d)
    (1, 1, 2, 128, 128)
    >>> _partition_shape_for_chunking((1, 2, 1, 32, 32), ds_shape_5d)
    (1, 2, 64, 128, 128)
    >>> _partition_shape_for_chunking((2, 1, 1, 32, 32), ds_shape_5d)
    (2, 64, 64, 128, 128)
    """
    first_non_one = [x == 1 for x in chunks].index(False)
    shape_left = chunks[:first_non_one + 1]
    return shape_left + ds_shape[first_non_one + 1:]


def _tileshape_for_chunking(chunks, ds_shape):
    """
    Calculate a tileshape for tiled reading from chunked
    data sets.

    Examples
    --------
    >>> ds_shape = Shape((64, 64, 128, 128), sig_dims=2)
    >>> _tileshape_for_chunking((1, 4, 32, 32), ds_shape)
    (4, 32, 32)
    """
    return chunks[-ds_shape.sig.dims - 1:]


def _get_tileshape_nd(partition_slice, tiling_scheme):
    extra_nav_dims = partition_slice.shape.nav.dims - tiling_scheme.shape.nav.dims
    # keep shape of the rightmost dimension:
    nav_item = min(tiling_scheme.shape[0], partition_slice.shape.nav[-1])
    return extra_nav_dims * (1,) + (nav_item,) + tuple(tiling_scheme.shape.sig)


class H5Reader:
    def __init__(self, path, ds_path):
        self._path = path
        self._ds_path = ds_path

    @contextlib.contextmanager
    def get_h5ds(self, cache_size=1024 * 1024):
        logger.debug("H5Reader.get_h5ds: cache_size=%dMiB", cache_size / 1024 / 1024)
        with h5py.File(self._path, 'r', rdcc_nbytes=cache_size, rdcc_nslots=19997) as f:
            yield f[self._ds_path]


class H5DataSet(DataSet):
    """
    Read data from a HDF5 data set.

    Examples
    --------

    >>> ds = ctx.load("hdf5", path=path_to_hdf5, ds_path="/data")

    Parameters
    ----------
    path: str
        Path to the file

    ds_path: str
        Path to the HDF5 data set inside the file

    nav_shape: tuple of int, optional
        A n-tuple that specifies the shape of the navigation / scan grid.
        By default this is inferred from the HDF5 dataset.

    sig_shape: tuple of int, optional
        A n-tuple that specifies the shape of the signal / frame grid.
        This parameter is currently unsupported and will raise an error
        if provided and not matching the underlying data sig shape.
        By default the sig_shape is inferred from the HDF5 dataset
        via the :code:`sig_dims` parameter.

    sig_dims: int
        Number of dimensions that should be considered part of the signal (for example
        2 when dealing with 2D image data)

    sync_offset: int, optional, by default 0
        If positive, number of frames to skip from start
        If negative, number of blank frames to insert at start

    target_size: int
        Target partition size, in bytes. Usually doesn't need to be changed.

    min_num_partitions: int
        Minimum number of partitions, set to number of cores if not specified. Usually
        doesn't need to be specified.

    Note
    ----
    If the HDF5 file to be loaded contains compressed data
    using a custom compression filter (other than GZIP, LZF or SZIP),
    the associated HDF5 filter library must be imported on the workers
    before accessing the file. See the `h5py documentation
    on filter pipelines
    <https://docs.h5py.org/en/stable/high/dataset.html#filter-pipeline>`_
    for more information.

    The library `hdf5plugin <https://github.com/silx-kit/hdf5plugin>`_
    is preloaded automatically if it is installed. Other filter libraries
    may have to be specified for preloading by the user.

    Preloads for a local :class:`~libertem.executor.dask.DaskJobExecutor` can be
    specified through the :code:`preload` argument of either
    :meth:`~libertem.executor.dask.DaskJobExecutor.make_local` or
    :func:`libertem.executor.dask.cluster_spec`. For the
    :class:`libertem.executor.inline.InlineJobExecutor`, the plugins
    can simply be imported in the main script.

    For the web GUI or for running LiberTEM in a cluster with existing workers
    (e.g. by running
    :code:`libertem-worker` or :code:`dask-worker` on nodes),
    necessary imports can be specified as :code:`--preload` arguments to
    the launch command,
    for example with :code:`libertem-server --preload hdf5plugin` resp.
    :code:`libertem-worker --preload hdf5plugin tcp://scheduler_ip:port`.
    :code:`--preload` can be specified multiple times.
    """
    def __init__(self, path, ds_path=None, tileshape=None, nav_shape=None, sig_shape=None,
                 target_size=None, min_num_partitions=None, sig_dims=2, io_backend=None,
                 sync_offset: int = 0):
        super().__init__(io_backend=io_backend)
        if io_backend is not None:
            raise ValueError("H5DataSet currently doesn't support alternative I/O backends")
        self.path = path
        self.ds_path = ds_path
        self.target_size = target_size
        self.sig_dims = sig_dims
        # handle backwards-compatability:
        if tileshape is not None:
            warnings.warn(
                "tileshape argument is ignored and will be removed after 0.6.0",
                FutureWarning
            )
        # self.min_num_partitions appears to be never used
        self.min_num_partitions = min_num_partitions
        self._dtype = None
        self._shape = None
        self._nav_shape = nav_shape
        self._sig_shape = sig_shape
        self._sync_offset = sync_offset
        self._chunks = None
        self._compression = None

    def get_reader(self):
        return H5Reader(
            path=self.path,
            ds_path=self.ds_path
        )

    def _do_initialize(self):
        if self.ds_path is None:
            try:
                datasets = _get_datasets(self.path)
                largest_ds = max(datasets, key=lambda x: prod(x.shape))
            # FIXME: excepting `SystemError` temporarily
            # more info: https://github.com/h5py/h5py/issues/1740
            except (ValueError, TimeoutError, SystemError):
                raise DataSetException(f'Unable to infer dataset from file {self.path}')
            self.ds_path = largest_ds.name
        with self.get_reader().get_h5ds() as h5ds:
            self._dtype = h5ds.dtype
            shape = h5ds.shape
            if len(shape) == self.sig_dims:
                # shape = (1,) + shape -> this leads to indexing errors down the line
                # so we currently don't support opening 2D HDF5 files
                raise DataSetException("2D HDF5 files are currently not supported")
            ds_shape = Shape(shape, sig_dims=self.sig_dims)
            if self._sig_shape is not None and tuple(self._sig_shape) != ds_shape.sig.to_tuple():
                raise DataSetException("sig reshaping currently not supported with HDF5 files")
            self._image_count = ds_shape.nav.size
            if self._nav_shape is None:
                nav_shape = ds_shape.nav.to_tuple()
            else:
                nav_shape = tuple(self._nav_shape)
            self._shape = nav_shape + ds_shape.sig
            self._meta = DataSetMeta(
                shape=self.shape,
                raw_dtype=self._dtype,
                sync_offset=self._sync_offset,
                image_count=self._image_count,
                metadata={'ds_raw_shape': ds_shape}
            )
            self._chunks = h5ds.chunks
            self._compression = h5ds.compression
            if self._compression is not None:
                warnings.warn(
                    "Loading compressed HDF5, performance can be worse than with other formats",
                    RuntimeWarning
                )
            self._nav_shape_product = self._shape.nav.size
            self._sync_offset_info = self.get_sync_offset_info()
        return self

    def initialize(self, executor):
        return executor.run_function(self._do_initialize)

    @classmethod
    def get_msg_converter(cls):
        return HDF5DatasetParams

    @classmethod
    def get_supported_extensions(cls):
        return {"h5", "hdf5", "hspy", "nxs"}

    @classmethod
    def get_supported_io_backends(self):
        return []

    @classmethod
    def _do_detect(cls, path):
        try:
            with h5py.File(path, 'r'):
                pass
        except OSError as e:
            raise DataSetException(repr(e)) from e

    @classmethod
    def detect_params(cls, path, executor):
        try:
            executor.run_function(cls._do_detect, path)
        except (OSError, KeyError, ValueError, TypeError, DataSetException):
            # not a h5py file or can't open for some reason:
            return False

        # Read the dataset info from the file
        try:
            datasets = executor.run_function(_get_datasets, path)
            if not datasets:
                raise RuntimeError(f'Found no compatible datasets in the file {path}')
        # FIXME: excepting `SystemError` temporarily
        # more info: https://github.com/h5py/h5py/issues/1740
        except (RuntimeError, TimeoutError, SystemError):
            return {
                "parameters": {
                    "path": path,
                },
                "info": {
                    "datasets": [],
                }
            }

        # datasets contains at least one HDF5ArrayDescriptor
        # sig_dims is implicitly two here (for web GUI)
        sig_dims = 2
        full_info = [
            {
                "path": ds_item.name,
                "shape": ds_item.shape,
                "compression": ds_item.compression,
                "chunks": ds_item.chunks,
                "raw_nav_shape": ds_item.shape[:-sig_dims],
                "nav_shape": _ensure_2d_nav(ds_item.shape[:-sig_dims]),
                "sig_shape": ds_item.shape[-sig_dims:],
                "image_count": prod(ds_item.shape[:-sig_dims]),
            } for ds_item in datasets
        ]

        # use the largest size array as initial hdf5 dataset path
        # need to get info dict to access unpacked nav/sig shape
        # next line implements argmax on ds_descriptor.size
        ds_idx, _ = max(enumerate(datasets), key=lambda idx_x: prod(idx_x[1].shape))
        largest_ds = full_info[ds_idx]
        return {
            "parameters": {
                "path": path,
                "ds_path": largest_ds['path'],
                "nav_shape": largest_ds['nav_shape'],
                "sig_shape": largest_ds['sig_shape'],
            },
            "info": {
                "datasets": full_info
            }
        }

    @property
    def dtype(self):
        if self._dtype is None:
            raise RuntimeError("please call initialize")
        return self._dtype

    @property
    def shape(self):
        if self._shape is None:
            raise RuntimeError("please call initialize")
        return self._shape

    def check_valid(self):
        try:
            with self.get_reader().get_h5ds() as h5ds:
                h5ds.shape
            return True
        except (OSError, KeyError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_cache_key(self):
        return {
            "path": self.path,
            "ds_path": self.ds_path,
        }

    def get_diagnostics(self):
        with self.get_reader().get_h5ds() as ds:
            try:
                datasets = _get_datasets(self.path)
            # FIXME: excepting `SystemError` temporarily
            # more info: https://github.com/h5py/h5py/issues/1740
            except (TimeoutError, SystemError):
                datasets = []
            datasets = [
                {"name": descriptor.name, "value": [
                    {"name": "Size", "value": str(prod(descriptor.shape))},
                    {"name": "Shape", "value": str(descriptor.shape)},
                    {"name": "Datatype", "value": str(descriptor.dtype)},
                ]}
                for descriptor in sorted(
                    datasets, key=lambda i: prod(i.shape), reverse=True
                )
            ]
            return [
                {"name": "dtype", "value": str(ds.dtype)},
                {"name": "chunks", "value": str(ds.chunks)},
                {"name": "compression", "value": str(ds.compression)},
                {"name": "datasets", "value": datasets},
            ]

    def get_min_sig_size(self):
        if self._chunks is not None:
            return 1024  # allow for tiled processing w/ small-ish chunks
        # un-chunked HDF5 seems to prefer larger signal slices, so we aim for 32 4k blocks:
        return 32 * 4096 // np.dtype(self.meta.raw_dtype).itemsize

    def get_max_io_size(self) -> Optional[int]:
        if self._chunks is not None:
            # this may result in larger tile depth than necessary, but
            # it needs to be so big to pass the validation of the Negotiator. The tiles
            # won't ever be as large as the scheme dictates, anyway.
            # We limit it here to 256e6 elements, to also keep the chunk cache
            # usage reasonable:
            return int(256e6)
        return None  # use default value from Negotiator

    def get_base_shape(self, roi):
        if roi is not None:
            return (1,) + self.shape.sig
        if self._chunks is not None:
            sig_chunks = self._chunks[-self.shape.sig.dims:]
            return (1,) + sig_chunks
        return (1, 1,) + (self.shape[-1],)

    def adjust_tileshape(self, tileshape, roi):
        chunks = self._chunks
        sig_shape = self.shape.sig
        if roi is not None:
            return (1,) + sig_shape
        if chunks is not None and not _have_contig_chunks(chunks, self.shape):
            sig_chunks = chunks[-sig_shape.dims:]
            sig_ts = tileshape[-sig_shape.dims:]
            # if larger signal chunking is requested in the negotiation,
            # switch to full frames:
            if any(t > c for t, c in zip(sig_ts, sig_chunks)):
                # try to keep total tileshape size:
                tileshape_size = prod(tileshape)
                depth = max(1, tileshape_size // sig_shape.size)
                return (depth,) + sig_shape
            else:
                # depth needs to be limited to prod(chunks.nav)
                return _tileshape_for_chunking(chunks, self.shape)
        return tileshape

    def need_decode(self, roi, read_dtype, corrections):
        return True

    def get_partitions(self):
        # ds_shape = Shape(self.shape, sig_dims=self.sig_dims)
        ds_shape: Shape = self.meta['ds_raw_shape']
        ds_slice = Slice(origin=[0] * len(ds_shape), shape=ds_shape)
        target_size = self.target_size
        if target_size is None:
            if self._compression is None:
                target_size = 512 * 1024 * 1024
            else:
                target_size = 256 * 1024 * 1024
        partition_shape = self.partition_shape(
            target_size=target_size,
            dtype=self.dtype,
            containing_shape=ds_shape,
        ) + tuple(ds_shape.sig)

        # if the data is chunked in the navigation axes, choose a compatible
        # partition size (even important for non-compressed data!)
        chunks = self._chunks
        if chunks is not None and not _have_contig_chunks(chunks, ds_shape):
            partition_shape = _partition_shape_for_chunking(chunks, ds_shape)

        # -ve sync offset insert blank at beginning (skips at end)
        # +ve sync offset skips frames at beginning (blank at end)
        sync_offset = self._sync_offset
        ds_flat_shape = self.shape.flatten_nav()

        for slice_nd in ds_slice.subslices(partition_shape):
            raw_frames_slice = slice_nd.flatten_nav(ds_shape)
            raw_origin = raw_frames_slice.origin
            if self._sync_offset <= 0:
                # negative or zero s-o, shift right, clip length at end
                raw_frames_slice.origin = (raw_origin[0] + abs(sync_offset),) + raw_origin[1:]
            else:
                # positive s-o, shift left, clip part length at beginning
                corrected_nav_origin = raw_origin[0] - sync_offset
                if corrected_nav_origin < 0:
                    corrected_nav_size = raw_frames_slice.shape[0] + corrected_nav_origin
                    if corrected_nav_size <= 0:
                        # Empty partition, skip
                        continue
                    raw_frames_slice.shape = (corrected_nav_size,) + raw_frames_slice.shape.sig
                raw_frames_slice.origin = (max(0, corrected_nav_origin),) + raw_origin[1:]

            # All raw_frames_slice should have non-zero dims here
            partition_slice = raw_frames_slice.clip_to(ds_flat_shape)
            if any(v <= 0 for v in partition_slice.shape):
                # Empty partition after clip to desired shape, skip
                continue

            yield H5Partition(
                meta=self._meta,
                reader=self.get_reader(),
                partition_slice=partition_slice,
                slice_nd=slice_nd,
                io_backend=self.get_io_backend(),
                chunks=self._chunks,
                decoder=None,
                sync_offset=self._sync_offset,
            )

    def __repr__(self):
        return f"<H5DataSet of {self._dtype} shape={self._shape}>"


class H5Partition(Partition):
    def __init__(self, reader: H5Reader, slice_nd: Slice, chunks, sync_offset=0, *args, **kwargs):
        self.reader = reader
        self.slice_nd = slice_nd
        self._corrections = None
        self._chunks = chunks
        self._sync_offset = sync_offset
        super().__init__(*args, **kwargs)

    def _have_compatible_chunking(self):
        chunks = self._chunks
        if chunks is None:
            return True
        # all-1 in nav dims works:
        nav_dims = self.slice_nd.shape.nav.dims
        chunks_nav = self._chunks[:nav_dims]
        if all(c == 1 for c in chunks_nav):
            return True
        # everything else is problematic and needs special case:
        return False

    def _get_subslices_chunked_full_frame(self, scheme_lookup, nav_dims, tileshape_nd):
        """
        chunked full-frame reading. outer loop goes over the
        navigation coords of each chunk, inner loop is pushed into
        hdf5 by reading full frames need to order slices in a way that
        efficiently uses the chunk cache.
        """
        chunks_nav = self._chunks[:nav_dims]
        chunk_full_frame = chunks_nav + self.slice_nd.shape.sig
        chunk_slices = self.slice_nd.subslices(shape=chunk_full_frame)
        logger.debug(
            "_get_subslices_chunked_full_frame: chunking first by %r, then %r",
            chunk_full_frame, tileshape_nd,
        )
        for chunk_slice in chunk_slices:
            subslices = chunk_slice.subslices(shape=tileshape_nd)
            for subslice in subslices:
                idx = scheme_lookup[(subslice.origin[nav_dims:], subslice.shape[nav_dims:])]
                yield idx, subslice

    def _get_subslices_chunked_tiled(self, tiling_scheme, scheme_lookup, nav_dims, tileshape_nd):
        """
        general tiled reading w/ chunking outer loop is a chunk in
        signal dimensions, inner loop is over "rows in nav"
        """
        slice_nd_sig = self.slice_nd.sig
        slice_nd_nav = self.slice_nd.nav
        chunks_nav = self._chunks[:nav_dims]
        sig_slices = slice_nd_sig.subslices(tiling_scheme.shape.sig)
        logger.debug(
            "_get_subslices_chunked_tiled: chunking first by sig %r, then nav %r, finally %r",
            tiling_scheme.shape.sig, chunks_nav, tileshape_nd
        )
        for sig_slice in sig_slices:
            chunk_slices = slice_nd_nav.subslices(shape=chunks_nav)
            for chunk_slice_nav in chunk_slices:
                chunk_slice = Slice(
                    origin=chunk_slice_nav.origin + sig_slice.origin,
                    shape=chunk_slice_nav.shape + tuple(sig_slice.shape),
                )
                subslices = chunk_slice.subslices(shape=tileshape_nd)
                for subslice in subslices:
                    scheme_key = (subslice.origin[nav_dims:], subslice.shape[nav_dims:])
                    idx = scheme_lookup[scheme_key]
                    yield idx, subslice

    def _get_subslices(self, tiling_scheme):
        """
        Generate partition subslices for the given tiling scheme for the different cases.
        """
        if tiling_scheme.intent == "partition":
            tileshape_nd = self.slice_nd.shape
        else:
            tileshape_nd = _get_tileshape_nd(self.slice_nd, tiling_scheme)

        assert all(ts <= ps for (ts, ps) in zip(tileshape_nd, self.slice_nd.shape))

        nav_dims = self.slice_nd.shape.nav.dims

        # Three cases need to be handled:
        if self._have_compatible_chunking():
            # 1) no chunking, or compatible chunking. we are free to use
            #    whatever access pattern we deem efficient:
            logger.debug("using simple tileshape_nd slicing")
            subslices = self.slice_nd.subslices(shape=tileshape_nd)
            scheme_len = len(tiling_scheme)
            for idx, subslice in enumerate(subslices):
                scheme_idx = idx % scheme_len
                yield scheme_idx, subslice
        else:
            scheme_lookup = {
                (s.discard_nav().origin, tuple(s.discard_nav().shape)): idx
                for idx, s in tiling_scheme.slices
            }
            if len(tiling_scheme) == 1:
                logger.debug("using full-frame subslicing")
                yield from self._get_subslices_chunked_full_frame(
                    scheme_lookup, nav_dims, tileshape_nd
                )
            else:
                logger.debug("using chunk-adaptive subslicing")
                yield from self._get_subslices_chunked_tiled(
                    tiling_scheme, scheme_lookup, nav_dims, tileshape_nd
                )

    def _preprocess(self, tile_data, tile_slice):
        if self._corrections is None:
            return
        self._corrections.apply(tile_data, tile_slice)

    def _get_read_cache_size(self) -> float:
        chunks = self._chunks
        if chunks is None:
            return 1 * 1024 * 1024
        else:
            # heuristic on maximum chunk cache size based on number of cores
            # of the node this worker is running on, available memory, ...
            import psutil
            mem = psutil.virtual_memory()
            num_cores = psutil.cpu_count(logical=False)
            available: int = mem.available
            if num_cores is None:
                num_cores = 2
            cache_size: float = max(256 * 1024 * 1024, available * 0.8 / num_cores)
            return cache_size

    def _get_h5ds(self):
        cache_size = self._get_read_cache_size()
        return self.reader.get_h5ds(cache_size=cache_size)

    def _get_tiles_normal(self, tiling_scheme: TilingScheme, dest_dtype):
        with self._get_h5ds() as dataset:
            # because the dtype conversion done by HDF5 itself can be quite slow,
            # we need to use a buffer for reading in hdf5 native dtype:
            data_flat = np.zeros(tiling_scheme.shape, dtype=dataset.dtype).reshape((-1,))

            # ... and additionally a result buffer, for re-using the array used in the DataTile:
            data_flat_res = np.zeros(tiling_scheme.shape, dtype=dest_dtype).reshape((-1,))

            subslices = self._get_subslices(
                tiling_scheme=tiling_scheme,
            )

            sync_offset = self._sync_offset
            ds_num_frames = self.meta.shape.nav.size

            for scheme_idx, tile_slice in subslices:
                tile_slice_flat: Slice = tile_slice.flatten_nav(self.meta['ds_raw_shape'])
                raw_origin = tile_slice_flat.origin
                raw_shape = tile_slice_flat.shape

                # The following block translates from tile_slice in the raw array
                # to the partition coordinate system with sync_offset applied
                # By doing this before reading we can avoid reading some tiles
                # at the beginning/end of the dataset
                # We will still read tiles which partially overlap the nav space
                # and afterwards we drop the unecessary frames
                corrected_nav_origin = raw_origin[0] - sync_offset
                if corrected_nav_origin < 0:
                    # positive sync_offset, drop frames at beginning of DS
                    if abs(corrected_nav_origin) > tile_slice_flat.shape[0]:
                        # tile is completely before the first partition, can skip it
                        continue
                    # Clip at the beginning so adjust the tile shape
                    new_nav_size = tile_slice_flat.shape[0] + corrected_nav_origin
                    tile_slice_flat.shape = (new_nav_size,) + tile_slice_flat.shape.sig
                # Apply max(0, corrected_nav_origin) so we never provide negative nav coord
                tile_slice_flat.origin = (max(0, corrected_nav_origin),) + raw_origin[1:]
                # Now check for clipping at the end of the dataset
                final_frame_idx = tile_slice_flat.origin[0] + tile_slice_flat.shape[0]
                frames_beyond_end = final_frame_idx - ds_num_frames
                # We want to skip any tiles which are completely past the end of the dataset
                # and clip those which are only partially overlapping the final partition
                if frames_beyond_end >= tile_slice_flat.shape[0]:
                    # Empty tile after clip, skip
                    continue
                elif frames_beyond_end > 0:
                    # tile partially overlaps end of dataset, adjust the shape
                    new_nav_size = tile_slice_flat.shape[0] - frames_beyond_end
                    tile_slice_flat.shape = (new_nav_size,) + tile_slice_flat.shape.sig

                # Read the data in this block
                # cut buffer into the right size
                buf_size = tile_slice.shape.size
                buf = data_flat[:buf_size].reshape(tile_slice.shape)
                buf_res = data_flat_res[:buf_size].reshape(tile_slice.shape)
                dataset.read_direct(buf, source_sel=tile_slice.get())
                buf_res[:] = buf  # extra copy for faster dtype/endianess conversion
                tile_data = buf_res.reshape(raw_shape)

                # If the true tile origin is before the start of the dataset, must drop frames
                # This corresponds to the first raw tile which overlaps the first partition
                # and can only occur when sync_offset > 0
                if corrected_nav_origin < 0:
                    tile_data = tile_data[abs(corrected_nav_origin):, ...]

                # The final tiles in the dataset can partially overlap the final partition
                # Drop frames at end to match the partition size
                # we already verified if any frames will remain
                # this can occur for both +ve and -ve sync_offset
                if frames_beyond_end > 0:
                    tile_data = tile_data[:-frames_beyond_end, ...]

                # NOTE could the above two blocks ever apply simultaneously?
                # would the two operations conflict ?

                self._preprocess(tile_data, tile_slice_flat)
                yield DataTile(
                    tile_data,
                    tile_slice=tile_slice_flat,
                    scheme_idx=scheme_idx,
                )

    def _get_tiles_with_roi(self, roi, dest_dtype, tiling_scheme):
        # we currently don't chop up the frames when reading with a roi, so
        # the tiling scheme also must not contain more than one slice:
        # NOTE Why is this ??
        assert len(tiling_scheme) == 1, "incompatible tiling scheme! (%r)" % (tiling_scheme)

        flat_roi_nonzero = flat_nonzero(roi)
        start_at_frame = self.slice.origin[0]
        stop_at_frame = start_at_frame + self.slice.shape[0]
        part_mask = np.logical_and(flat_roi_nonzero >= start_at_frame,
                                   flat_roi_nonzero < stop_at_frame)
        # Must yield tiles with tile_slice in compressed nav dimension for roi
        frames_in_c_nav = np.arange(flat_roi_nonzero.size)[part_mask]
        frames_in_part = flat_roi_nonzero[part_mask]
        # -ve sync offset insert blank at beginning (skips at end)
        # +ve sync offset skips frames at beginning (blank at end)
        frames_in_raw = frames_in_part + self._sync_offset
        raw_shape = self.meta['ds_raw_shape'].nav.to_tuple()
        result_shape = Shape((1,) + tuple(self.meta.shape.sig), sig_dims=self.meta.shape.sig.dims)
        sig_origin = (0,) * self.meta.shape.sig.dims
        tile_data = np.zeros(result_shape, dtype=dest_dtype)

        with self._get_h5ds() as h5ds:
            tile_data_raw = np.zeros(result_shape, dtype=h5ds.dtype)
            for c_nav_idx, raw_idx in zip(frames_in_c_nav, frames_in_raw):
                tile_slice = Slice(
                    origin=(c_nav_idx,) + sig_origin,
                    shape=result_shape,
                )
                nav_coord = np.unravel_index(raw_idx, raw_shape)
                h5ds.read_direct(tile_data_raw, source_sel=nav_coord)
                tile_data[:] = tile_data_raw  # extra copy for dtype/endianess conversion
                self._preprocess(tile_data, tile_slice)
                yield DataTile(
                    tile_data,
                    tile_slice=tile_slice,
                    # there is only a single slice in the tiling scheme, so our
                    # scheme_idx is constant 0
                    scheme_idx=0,
                )

    def set_corrections(self, corrections: CorrectionSet):
        self._corrections = corrections

    def get_tiles(self, tiling_scheme: TilingScheme, dest_dtype="float32", roi=None,
            array_backend: Optional[ArrayBackend] = None):
        if array_backend is None:
            array_backend = self.meta.array_backends[0]
        assert array_backend in (NUMPY, CUDA)
        tiling_scheme = tiling_scheme.adjust_for_partition(self)
        if roi is not None:
            yield from self._get_tiles_with_roi(roi, dest_dtype, tiling_scheme)
        else:
            yield from self._get_tiles_normal(tiling_scheme, dest_dtype)

    def get_locations(self):
        return None

    def get_macrotile(self, dest_dtype="float32", roi=None):
        '''
        Return a single tile for the entire partition.

        This is useful to support process_partiton() in UDFs and to construct
        dask arrays from datasets.

        Note
        ----

        This can be inefficient if the dataset is compressed and chunked in the
        navigation axis, because you can either have forced-large macrotiles,
        or you can have read amplification effects, where a much larger amount
        of data is read from the HDF5 file than necessary.

        For example, if your chunking is :code:`(32, 32, 32, 32)`, in a
        dataset that is :code:`(128, 128, 256, 256)`, the partition must
        cover the whole of :code:`(32, 128, 256, 256)` - this is because
        partitions are contiguous in the navigation axis.

        The other possibility is to keep the partition smaller, for example
        only :code:`(3, 128, 256, 256)`. That would mean when reading a chunk
        from HDF5, we can only use 3*32 frames of the total 32*32 frames,
        a whopping ~10x read amplification.
        '''

        tileshape = self.shape
        if self._chunks is not None:
            tileshape = self._chunks

        tiling_scheme = TilingScheme.make_for_shape(
            tileshape=Shape(tileshape, sig_dims=self.slice.shape.sig.dims).flatten_nav(),
            dataset_shape=self.meta.shape,
        )

        data = zeros_aligned(self.slice.adjust_for_roi(roi).shape, dtype=dest_dtype)

        for tile in self.get_tiles(
            tiling_scheme=tiling_scheme,
            dest_dtype=dest_dtype,
            roi=roi,
        ):
            rel_slice = tile.tile_slice.shift(self.slice)
            data[rel_slice.get()] = tile.data
        tile_slice = Slice(
            origin=(self.slice.origin[0], 0, 0),
            shape=Shape(data.shape, sig_dims=2),
        )
        return DataTile(
            data,
            tile_slice=tile_slice,
            scheme_idx=0,
        )
