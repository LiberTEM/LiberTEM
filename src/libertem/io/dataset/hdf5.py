import contextlib
import warnings
import time

import numpy as np
import h5py

from libertem.common import Slice, Shape
from libertem.corrections import CorrectionSet
from libertem.web.messages import MessageConverter
from .base import (
    DataSet, Partition, DataTile, DataSetException, DataSetMeta, _roi_to_nd_indices,
    TilingScheme,
)


# alias for mocking:
current_time = time.time


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
            datasets.append((name, obj.size, obj.shape, obj.dtype))

    with h5py.File(path, 'r') as f:
        f.visititems(_make_list)
    return datasets


class H5Reader(object):
    def __init__(self, path, ds_path):
        self._path = path
        self._ds_path = ds_path

    @contextlib.contextmanager
    def get_h5ds(self):
        with h5py.File(self._path, 'r') as f:
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
                 target_size=512*1024*1024, min_num_partitions=None, sig_dims=2, io_backend=None):
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

    def get_reader(self):
        return H5Reader(
            path=self.path,
            ds_path=self.ds_path
        )

    def _do_initialize(self):
        if self.ds_path is None:
            datasets = _get_datasets(self.path)
            datasets_list = sorted(datasets, key=lambda i: i[1], reverse=True)
            name, size, shape, dtype = datasets_list[0]
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
            datasets_list = sorted(datasets, key=lambda i: i[1], reverse=True)
            dataset_paths = [ds_path[0] for ds_path in datasets_list]
            name, size, shape, dtype = datasets_list[0]
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
                "dataset_paths": dataset_paths,
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
                for name, size, shape, dtype in sorted(
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
        partition_shape = self.partition_shape(
            target_size=self.target_size,
            dtype=self.dtype,
        ) + tuple(self.shape.sig)
        for pslice in ds_slice.subslices(partition_shape):
            yield H5Partition(
                meta=self._meta,
                reader=self.get_reader(),
                partition_slice=pslice.flatten_nav(self.shape),
                slice_nd=pslice,
                io_backend=self.get_io_backend(),
            )

    def __repr__(self):
        return "<H5DataSet of %s shape=%s>" % (self._dtype, self._shape)


class H5Partition(Partition):
    def __init__(self, reader, slice_nd, *args, **kwargs):
        self.reader = reader
        self.slice_nd = slice_nd
        self._corrections = None
        super().__init__(*args, **kwargs)

    def _get_subslices(self, tiling_scheme, tileshape_nd):
        subslices = self.slice_nd.subslices(shape=tileshape_nd)
        scheme_len = len(tiling_scheme)
        for idx, subslice in enumerate(subslices):
            scheme_idx = idx % scheme_len
            yield scheme_idx, subslice

    def _preprocess(self, tile_data, tile_slice):
        if self._corrections is None:
            return
        self._corrections.apply(tile_data, tile_slice)

    def _get_tiles_normal(self, tiling_scheme, tileshape_nd, dest_dtype="float32"):
        data = np.zeros(tileshape_nd, dtype=dest_dtype)
        with self.reader.get_h5ds() as dataset:
            subslices = self._get_subslices(
                tiling_scheme=tiling_scheme,
                tileshape_nd=tileshape_nd,
            )
            for scheme_idx, tile_slice in subslices:
                tile_slice_flat = tile_slice.flatten_nav(self.meta.shape)
                if tuple(tile_slice.shape) != tuple(tileshape_nd):
                    # at the border, can't reuse buffer
                    border_data = np.zeros(tile_slice.shape, dtype=dest_dtype)
                    buf = border_data
                else:
                    # reuse buffer
                    buf = data
                dataset.read_direct(buf, source_sel=tile_slice.get())
                tile_data = buf.reshape(tile_slice_flat.shape)
                self._preprocess(tile_data, tile_slice_flat)
                yield DataTile(
                    tile_data,
                    tile_slice=tile_slice_flat,
                    scheme_idx=scheme_idx,
                )

    def _get_tiles_with_roi(self, roi, dest_dtype):
        flat_roi = roi.reshape((-1,))
        roi = roi.reshape(self.meta.shape.nav)

        result_shape = Shape((1,) + tuple(self.meta.shape.sig), sig_dims=self.meta.shape.sig.dims)
        sig_origin = tuple([0] * self.meta.shape.sig.dims)
        frames_read = 0
        start_at_frame = self.slice.origin[0]
        frame_offset = np.count_nonzero(flat_roi[:start_at_frame])

        indices = _roi_to_nd_indices(roi, self.slice_nd)

        with self.reader.get_h5ds() as h5ds:
            for idx in indices:
                tile_slice = Slice(
                    origin=(frames_read + frame_offset,) + sig_origin,
                    shape=result_shape,
                )
                tile_data = h5ds[idx].reshape(result_shape)
                self._preprocess(tile_data, tile_slice)
                yield DataTile(
                    tile_data,
                    tile_slice=tile_slice,
                    # there is only a single slice in the tiling scheme, so our
                    # scheme_idx is constant 0
                    scheme_idx=0,
                )
                frames_read += 1

    def get_base_shape(self):
        with self.reader.get_h5ds() as h5ds:
            if h5ds.chunks is not None:
                return h5ds.chunks
        return (1, 1,) + (self.shape[-1],)

    def adjust_tileshape(self, tileshape):
        return tileshape

    def need_decode(self, roi, read_dtype, corrections):
        return True

    def set_corrections(self, corrections: CorrectionSet):
        self._corrections = corrections

    def get_tiles(self, tiling_scheme, dest_dtype="float32", roi=None):
        extra_nav_dims = self.meta.shape.nav.dims - tiling_scheme.shape.nav.dims
        tileshape_nd = extra_nav_dims * (1,) + tuple(tiling_scheme.shape)

        if roi is not None:
            yield from self._get_tiles_with_roi(roi, dest_dtype)
        else:
            yield from self._get_tiles_normal(tiling_scheme, tileshape_nd, dest_dtype)

    def get_locations(self):
        return None

    def get_macrotile(self, dest_dtype="float32", roi=None):
        '''
        Return a single tile for the entire partition.

        This is useful to support process_partiton() in UDFs and to construct dask arrays
        from datasets.
        '''

        tiling_scheme = TilingScheme.make_for_shape(
            tileshape=self.shape,
            dataset_shape=self.meta.shape,
        )

        try:
            return next(self.get_tiles(
                tiling_scheme=tiling_scheme,
                dest_dtype=dest_dtype,
                roi=roi,
            ))
        except StopIteration:
            tile_slice = Slice(
                origin=(self.slice.origin[0], 0, 0),
                shape=Shape((0,) + tuple(self.slice.shape.sig), sig_dims=2),
            )
            return DataTile(
                np.zeros(tile_slice.shape, dtype=dest_dtype),
                tile_slice=tile_slice,
                scheme_idx=0,
            )
