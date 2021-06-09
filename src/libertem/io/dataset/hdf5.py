import contextlib
import warnings
import logging
import time

import numpy as np
import h5py

from libertem.common import Slice, Shape
from libertem.common.buffers import zeros_aligned
from libertem.corrections import CorrectionSet
from libertem.web.messages import MessageConverter
from .base import (
    DataSet, Partition, DataTile, DataSetException, DataSetMeta, _roi_to_nd_indices,
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
        },
        "required": ["type", "path", "ds_path"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path", "ds_path"]
        }
        return data


def _get_datasets(path):
    datasets = []

    timeout = 3
    t0 = current_time()

    def _make_list(name, obj):
        if current_time() - t0 > timeout:
            raise TimeoutError
        if hasattr(obj, 'size') and hasattr(obj, 'shape'):
            datasets.append(
                (name, obj.size, obj.shape, obj.dtype, obj.compression, obj.chunks)
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
    # There exists an index `i` such that `np.prod(chunks[:i]) == 1` and
    # chunks[i+1:] == ds_shape[i+1:], limited to the nav part of chunks and ds_shape
    #
    nav_shape = tuple(ds_shape.nav)
    nav_dims = len(nav_shape)
    chunks_nav = chunks[:nav_dims]

    for i in range(nav_dims):
        left = chunks_nav[:i]
        left_prod = np.prod(left, dtype=np.uint64)
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


class H5Reader(object):
    def __init__(self, path, ds_path):
        self._path = path
        self._ds_path = ds_path

    @contextlib.contextmanager
    def get_h5ds(self, cache_size=1024*1024):
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

    sig_dims: int
        Number of dimensions that should be considered part of the signal (for example
        2 when dealing with 2D image data)

    target_size: int
        Target partition size, in bytes. Usually doesn't need to be changed.

    min_num_partitions: int
        Minimum number of partitions, set to number of cores if not specified. Usually
        doesn't need to be specified.
    """
    def __init__(self, path, ds_path=None, tileshape=None,
                 target_size=None, min_num_partitions=None, sig_dims=2, io_backend=None):
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
        self.min_num_partitions = min_num_partitions
        self._dtype = None
        self._shape = None
        self._sync_offset = 0
        self._chunks = None
        self._compression = None

    def get_reader(self):
        return H5Reader(
            path=self.path,
            ds_path=self.ds_path
        )

    def _do_initialize(self):
        if self.ds_path is None:
            datasets = _get_datasets(self.path)
            datasets_list = sorted(datasets, key=lambda i: i[1], reverse=True)
            name, size, shape, dtype, compression, chunks = datasets_list[0]
            self.ds_path = name
        with self.get_reader().get_h5ds() as h5ds:
            self._dtype = h5ds.dtype
            shape = h5ds.shape
            if len(shape) == self.sig_dims:
                # shape = (1,) + shape -> this leads to indexing errors down the line
                # so we currently don't support opening 2D HDF5 files
                raise DataSetException("2D HDF5 files are currently not supported")
            self._shape = Shape(shape, sig_dims=self.sig_dims)
            self._image_count = self._shape.nav.size
            self._meta = DataSetMeta(
                shape=self.shape,
                raw_dtype=self._dtype,
                sync_offset=self._sync_offset,
                image_count=self._image_count,
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
        return set(["h5", "hdf5", "hspy", "nxs"])

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
        except (IOError, OSError, KeyError, ValueError, TypeError, DataSetException):
            # not a h5py file or can't open for some reason:
            return False

        # try to guess the hdf5 dataset path:
        try:
            datasets = executor.run_function(_get_datasets, path)
            datasets_list = list(sorted(datasets, key=lambda i: i[1], reverse=True))
            name, size, shape, dtype, compression, chunks = datasets_list[0]
        # FIXME: excepting `SystemError` temporarily
        # more info: https://github.com/h5py/h5py/issues/1740
        except (IndexError, TimeoutError, SystemError):
            return {
                "parameters": {
                    "path": path,
                }
            }

        return {
            "parameters": {
                "path": path,
                "ds_path": name,
            },
            "info": {
                "datasets": [
                    {
                        "path": ds_item[0],
                        "shape": ds_item[2],
                        "compression": ds_item[4],
                        "chunks": ds_item[5],
                    }
                    for ds_item in datasets_list
                ]
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
        except (IOError, OSError, KeyError, ValueError) as e:
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
                {"name": name, "value": [
                    {"name": "Size", "value": str(size)},
                    {"name": "Shape", "value": str(shape)},
                    {"name": "Datatype", "value": str(dtype)},
                ]}
                for name, size, shape, dtype, compression, chunks in sorted(
                    datasets, key=lambda i: i[1], reverse=True
                )
            ]
            return [
                {"name": "dtype", "value": str(ds.dtype)},
                {"name": "chunks", "value": str(ds.chunks)},
                {"name": "compression", "value": str(ds.compression)},
                {"name": "datasets", "value": datasets},
            ]

    def get_partitions(self):
        ds_shape = Shape(self.shape, sig_dims=self.sig_dims)
        ds_slice = Slice(origin=[0] * len(self.shape), shape=ds_shape)
        target_size = self.target_size
        if target_size is None:
            if self._compression is None:
                target_size = 512*1024*1024
            else:
                target_size = 256*1024*1024
        partition_shape = self.partition_shape(
            target_size=target_size,
            dtype=self.dtype,
        ) + tuple(self.shape.sig)

        # if the data is chunked in the navigation axes, choose a compatible
        # partition size (even important for non-compressed data!)
        chunks = self._chunks
        if chunks is not None and not _have_contig_chunks(chunks, ds_shape):
            partition_shape = _partition_shape_for_chunking(chunks, ds_shape)

        for pslice in ds_slice.subslices(partition_shape):
            yield H5Partition(
                meta=self._meta,
                reader=self.get_reader(),
                partition_slice=pslice.flatten_nav(self.shape),
                slice_nd=pslice,
                io_backend=self.get_io_backend(),
                chunks=self._chunks,
            )

    def __repr__(self):
        return "<H5DataSet of %s shape=%s>" % (self._dtype, self._shape)


class H5Partition(Partition):
    def __init__(self, reader, slice_nd, chunks, *args, **kwargs):
        self.reader = reader
        self.slice_nd = slice_nd
        self._corrections = None
        self._chunks = chunks
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
        extra_nav_dims = self.meta.shape.nav.dims - tiling_scheme.shape.nav.dims
        tileshape_nd = extra_nav_dims * (1,) + tuple(tiling_scheme.shape)

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

    def _get_read_cache_size(self) -> int:
        chunks = self._chunks
        if chunks is None:
            return 1*1024*1024
        else:
            # heuristic on maximum chunk cache size based on number of cores
            # of the node this worker is running on, available memory, ...
            import psutil
            mem = psutil.virtual_memory()
            num_cores = psutil.cpu_count(logical=False)
            if num_cores is None:
                num_cores = 2
            return max(256*1024*1024, mem.available * 0.8 / num_cores)

    def _get_h5ds(self):
        cache_size = self._get_read_cache_size()
        return self.reader.get_h5ds(cache_size=cache_size)

    def _get_tiles_normal(self, tiling_scheme, dest_dtype):
        with self._get_h5ds() as dataset:
            # because the dtype conversion done by HDF5 itself can be quite slow,
            # we need to use a buffer for reading in hdf5 native dtype:
            data_flat = zeros_aligned(tiling_scheme.shape, dtype=dataset.dtype).reshape((-1,))

            # ... and additionally a result buffer, for re-using the array used in the DataTile:
            data_flat_res = zeros_aligned(tiling_scheme.shape, dtype=dest_dtype).reshape((-1,))

            subslices = self._get_subslices(
                tiling_scheme=tiling_scheme,
            )
            for scheme_idx, tile_slice in subslices:
                tile_slice_flat = tile_slice.flatten_nav(self.meta.shape)
                # cut buffer into the right size
                buf_size = tile_slice.shape.size
                buf = data_flat[:buf_size].reshape(tile_slice.shape)
                buf_res = data_flat_res[:buf_size].reshape(tile_slice.shape)
                dataset.read_direct(buf, source_sel=tile_slice.get())
                buf_res[:] = buf  # extra copy for faster dtype/endianess conversion
                tile_data = buf_res.reshape(tile_slice_flat.shape)
                self._preprocess(tile_data, tile_slice_flat)
                yield DataTile(
                    tile_data,
                    tile_slice=tile_slice_flat,
                    scheme_idx=scheme_idx,
                )

    def _get_tiles_with_roi(self, roi, dest_dtype, tiling_scheme):
        # we currently don't chop up the frames when reading with a roi, so
        # the tiling scheme also must not contain more than one slice:
        assert len(tiling_scheme) == 1, "incompatible tiling scheme! (%r)" % (tiling_scheme)

        flat_roi = roi.reshape((-1,))
        roi = roi.reshape(self.meta.shape.nav)

        result_shape = Shape((1,) + tuple(self.meta.shape.sig), sig_dims=self.meta.shape.sig.dims)
        sig_origin = tuple([0] * self.meta.shape.sig.dims)
        frames_read = 0
        start_at_frame = self.slice.origin[0]
        frame_offset = np.count_nonzero(flat_roi[:start_at_frame])

        indices = _roi_to_nd_indices(roi, self.slice_nd)

        tile_data = np.zeros(result_shape, dtype=dest_dtype)

        with self._get_h5ds() as h5ds:
            tile_data_raw = np.zeros(result_shape, dtype=h5ds.dtype)
            for idx in indices:
                tile_slice = Slice(
                    origin=(frames_read + frame_offset,) + sig_origin,
                    shape=result_shape,
                )
                h5ds.read_direct(tile_data_raw, source_sel=idx)
                tile_data[:] = tile_data_raw  # extra copy for dtype/endianess conversion
                self._preprocess(tile_data, tile_slice)
                yield DataTile(
                    tile_data,
                    tile_slice=tile_slice,
                    # there is only a single slice in the tiling scheme, so our
                    # scheme_idx is constant 0
                    scheme_idx=0,
                )
                frames_read += 1

    def get_base_shape(self, roi):
        if roi is not None:
            return (1,) + self.shape.sig
        if self._chunks is not None:
            sig_chunks = self._chunks[-self.shape.sig.dims:]
            return (1,) + sig_chunks
        return (1, 1,) + (self.shape[-1],)

    def get_min_sig_size(self):
        if self._chunks is not None:
            return 1024  # allow for tiled processing w/ small-ish chunks
        # un-chunked HDF5 seems to prefer larger signal slices, so we aim for 32 4k blocks:
        return 32 * 4096 // np.dtype(self.meta.raw_dtype).itemsize

    def adjust_tileshape(self, tileshape, roi):
        chunks = self._chunks
        if roi is not None:
            return (1,) + self.shape.sig
        if chunks is not None and not _have_contig_chunks(chunks, self.shape):
            sig_chunks = chunks[-self.shape.sig.dims:]
            sig_ts = tileshape[-self.shape.sig.dims:]
            # if larger signal chunking is requested in the negotiation,
            # switch to full frames:
            if any(t > c for t, c in zip(sig_ts, sig_chunks)):
                # try to keep total tileshape size:
                tileshape_size = np.prod(tileshape, dtype=np.int64)
                depth = max(1, tileshape_size // self.shape.sig.size)
                return (depth,) + self.shape.sig
            else:
                # depth needs to be limited to prod(chunks.nav)
                return _tileshape_for_chunking(chunks, self.meta.shape)
        return tileshape

    def need_decode(self, roi, read_dtype, corrections):
        return True

    def set_corrections(self, corrections: CorrectionSet):
        self._corrections = corrections

    def get_max_io_size(self):
        if self._chunks is not None:
            # this may result in larger tile depth than necessary, but
            # it needs to be so big to pass the validation of the Negotiator. The tiles
            # won't ever be as large as the scheme dictates, anyway.
            # We limit it here to 256e6 elements, to also keep the chunk cache
            # usage reasonable:
            return 256e6
        return None  # use default value from Negotiator

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
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
            data[rel_slice.get()] = tile
        tile_slice = Slice(
            origin=(self.slice.origin[0], 0, 0),
            shape=Shape(data.shape, sig_dims=2),
        )
        return DataTile(
            data,
            tile_slice=tile_slice,
            scheme_idx=0,
        )
