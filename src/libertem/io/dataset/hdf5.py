import contextlib
import time

import numpy as np
import h5py

from libertem.common import Slice, Shape
from libertem.io.utils import get_partition_shape
from libertem.web.messages import MessageConverter
from .base import (
    DataSet, Partition, DataTile, DataSetException, DataSetMeta, _roi_to_nd_indices
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
            "tileshape": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 4,
                "maxItems": 4,
            },
        },
        "required": ["type", "path", "ds_path", "tileshape"]
    }

    def convert_to_python(self, raw_data):
        data = {
            k: raw_data[k]
            for k in ["path", "ds_path", "tileshape"]
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

    >>> ds = ctx.load("hdf5", path=path_to_hdf5, ds_path="/data", tileshape=(5, 16, 16))

    Parameters
    ----------
    path: str
        Path to the file

    ds_path: str
        Path to the HDF5 data set inside the file

    tileshape: tuple of int
        Tuning parameter, specifying the size of the smallest data unit
        we are reading and working on. Should match dimensionality of the data set.

    sig_dims: int
        Number of dimensions that should be considered part of the signal (for example
        2 when dealing with 2D image data)

    target_size: int
        Target partition size, in bytes. Usually doesn't need to be changed.

    min_num_partitions: int
        Minimum number of partitions, set to number of cores if not specified. Usually
        doesn't need to be specified.
    """
    def __init__(self, path, ds_path, tileshape=None,
                 target_size=512*1024*1024, min_num_partitions=None, sig_dims=2):
        super().__init__()
        self.path = path
        self.ds_path = ds_path
        self.target_size = target_size
        self.sig_dims = sig_dims
        self.tileshape = None
        if tileshape is not None:
            self.tileshape = Shape(tileshape, sig_dims=self.sig_dims)
        self.min_num_partitions = min_num_partitions
        self._dtype = None
        self._shape = None

    def get_reader(self):
        return H5Reader(
            path=self.path,
            ds_path=self.ds_path
        )

    def _do_initialize(self):
        with self.get_reader().get_h5ds() as h5ds:
            self._dtype = h5ds.dtype
            self._shape = Shape(h5ds.shape, sig_dims=self.sig_dims)
            self._meta = DataSetMeta(
                shape=self.shape,
                raw_dtype=self._dtype,
                iocaps={"FULL_FRAMES"},
            )
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
        except (IOError, OSError, KeyError, ValueError, DataSetException):
            # not a h5py file or can't open for some reason:
            return False

        # try to guess the hdf5 dataset path:
        try:
            datasets = executor.run_function(_get_datasets, path)
            datasets_list = sorted(datasets, key=lambda i: i[1], reverse=True)
            dataset_paths = [ds_path[0] for ds_path in datasets_list]
            name, size, shape, dtype = datasets_list[0]
        except (IndexError, TimeoutError):
            return {
                "parameters": {
                    "path": path
                }
            }

        return {
            "parameters": {
                "path": path,
                "ds_path": name,
                # FIXME: number of frames may not match L3 size
                "tileshape": (1, 8,) + shape[2:],
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
            except TimeoutError:
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
                tileshape=self.tileshape,
                meta=self._meta,
                reader=self.get_reader(),
                partition_slice=pslice.flatten_nav(self.shape),
                slice_nd=pslice,
            )

    def __repr__(self):
        return "<H5DataSet of %s shape=%s>" % (self._dtype, self._shape)


class H5Partition(Partition):
    def __init__(self, tileshape, reader, slice_nd, *args, **kwargs):
        self.tileshape = tileshape
        self.reader = reader
        self.slice_nd = slice_nd
        super().__init__(*args, **kwargs)

    def _get_tileshape(self, dest_dtype, target_size=None):
        if self.tileshape is not None:
            return self.tileshape
        if target_size is None:
            target_size = 1 * 1024 * 1024
        nav_shape = get_partition_shape(
            dataset_shape=self.slice_nd.shape,
            target_size_items=target_size // np.dtype(dest_dtype).itemsize,
        )
        return Shape(
            nav_shape + tuple(self.slice_nd.shape.sig),
            sig_dims=self.slice_nd.shape.sig.dims,
        )

    def _get_tiles_normal(self, tileshape, crop_to=None, dest_dtype="float32"):
        if crop_to is not None:
            if crop_to.shape.sig != self.meta.shape.sig:
                raise DataSetException("H5DataSet only supports whole-frame crops for now")
        data = np.ndarray(tileshape, dtype=dest_dtype)
        with self.reader.get_h5ds() as dataset:
            subslices = self.slice_nd.subslices(shape=tileshape)
            for tile_slice in subslices:
                tile_slice_flat = tile_slice.flatten_nav(self.meta.shape)
                if crop_to is not None:
                    intersection = tile_slice_flat.intersection_with(crop_to)
                    if intersection.is_null():
                        continue
                if tuple(tile_slice.shape) != tuple(tileshape):
                    # at the border, can't reuse buffer
                    border_data = np.ndarray(tile_slice.shape, dtype=dest_dtype)
                    buf = border_data
                else:
                    # reuse buffer
                    buf = data
                dataset.read_direct(buf, source_sel=tile_slice.get())
                yield DataTile(data=buf.reshape(tile_slice_flat.shape), tile_slice=tile_slice_flat)

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
                yield DataTile(
                    data=h5ds[idx].reshape(result_shape),
                    tile_slice=tile_slice
                )
                frames_read += 1

    def get_tiles(self, crop_to=None, full_frames=False, mmap=False, dest_dtype="float32",
                  roi=None, target_size=None):
        if crop_to is not None and roi is not None:
            if crop_to.shape.nav.size != self._num_frames:
                raise ValueError("don't use crop_to with roi")
        tileshape = self._get_tileshape(dest_dtype, target_size)
        if full_frames:
            tileshape = (
                tuple(tileshape.nav) + tuple(self.meta.shape.sig)
            )
        if roi is not None:
            yield from self._get_tiles_with_roi(roi, dest_dtype)
        else:
            yield from self._get_tiles_normal(tileshape, crop_to, dest_dtype)
