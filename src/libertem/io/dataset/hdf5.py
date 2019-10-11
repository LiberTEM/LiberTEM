import contextlib
import time

import numpy as np
import h5py

from libertem.common import Slice, Shape
from libertem.web.messages import MessageConverter
from .base import DataSet, Partition, DataTile, DataSetException, DataSetMeta


def unravel_nav(slice_, containing_shape):
    """
    inverse of flatten_nav, currently limited to 1d -> 2d nav
    """
    sig_dims = slice_.shape.sig.dims
    nav_dims = slice_.shape.dims - sig_dims
    nav_origin = np.unravel_index(
        slice_.origin[0],
        tuple(containing_shape)[:-sig_dims]
    )
    assert nav_dims == 1, "unravel_nav only works if nav.dims is currently 1"
    assert (len(containing_shape) - len(slice_.shape)) == 1,\
        "currently only works for 1d -> 2d case"
    nav_shape = (slice_.shape[0] // containing_shape[1], containing_shape[1])
    return Slice(
        origin=nav_origin + slice_.origin[-sig_dims:],
        shape=Shape(nav_shape + tuple(slice_.shape.sig), sig_dims=sig_dims)
    )


class HDF5DatasetParams(MessageConverter):
    SCHEMA = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "$id": "http://libertem.org/HDF5DatasetParams.schema.json",
        "title": "HDF5DatasetParams",
        "type": "object",
        "properties": {
            "type": {"const": "hdf5"},
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
    t0 = time.time()

    def _make_list(name, obj):
        if time.time() - t0 > timeout:
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
    def __init__(self, path, ds_path, tileshape,
                 target_size=512*1024*1024, min_num_partitions=None, sig_dims=2):
        super().__init__()
        self.path = path
        self.ds_path = ds_path
        self.target_size = target_size
        self.sig_dims = sig_dims
        self.tileshape = Shape(tileshape, sig_dims=self.sig_dims)
        self.min_num_partitions = min_num_partitions
        self._dtype = None
        self._shape = None

    def get_reader(self):
        return H5Reader(
            path=self.path,
            ds_path=self.ds_path
        )

    def initialize(self):
        with self.get_reader().get_h5ds() as h5ds:
            self._dtype = h5ds.dtype
            self._shape = Shape(h5ds.shape, sig_dims=self.sig_dims)
            self._meta = DataSetMeta(
                shape=self.shape,
                raw_dtype=self._dtype,
                iocaps={"FULL_FRAMES"},
            )
        return self

    @classmethod
    def get_msg_converter(cls):
        return HDF5DatasetParams

    @classmethod
    def detect_params(cls, path):
        try:
            with h5py.File(path, 'r'):
                pass
        except (IOError, OSError, KeyError, ValueError):
            # not a h5py file or can't open for some reason:
            return False

        # try to guess the hdf5 dataset path:
        try:
            datasets = _get_datasets(path)
            largest_ds = sorted(datasets, key=lambda i: i[1], reverse=True)[0]
            name, size, shape, dtype = largest_ds
        except (IndexError, TimeoutError):
            return {"path": path}

        return {
            "path": path,
            "ds_path": name,
            # FIXME: number of frames may not match L3 size
            "tileshape": (1, 8,) + shape[2:],
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
                for name, size, shape, dtype in sorted(datasets, key=lambda i: i[0])
            ]
            return [
                {"name": "dtype", "value": str(ds.dtype)},
                {"name": "chunks", "value": str(ds.chunks)},
                {"name": "compression", "value": str(ds.compression)},
                {"name": "datasets", "value": datasets},
            ]

    def get_partitions(self):
        ds_shape = Shape(self.shape, sig_dims=self.sig_dims)
        ds_slice = Slice(origin=(0, 0, 0, 0), shape=ds_shape)
        dtype = self.dtype
        partition_shape = self.partition_shape(
            datashape=self.shape,
            framesize=self.shape.sig.size,
            dtype=dtype,
            target_size=self.target_size,
            min_num_partitions=self.min_num_partitions,
        )
        for pslice in ds_slice.subslices(partition_shape):
            # TODO: where should the tileshape be set? let the user choose for now
            yield H5Partition(
                tileshape=self.tileshape,
                meta=self._meta,
                reader=self.get_reader(),
                partition_slice=pslice.flatten_nav(self.shape),
            )

    def __repr__(self):
        return "<H5DataSet of %s shape=%s>" % (self._dtype, self._shape)


class H5Partition(Partition):
    def __init__(self, tileshape, reader, *args, **kwargs):
        self.tileshape = tileshape
        self.reader = reader
        super().__init__(*args, **kwargs)

    def _get_tiles_normal(self, tileshape, crop_to=None, dest_dtype="float32"):
        if crop_to is not None:
            if crop_to.shape.sig != self.meta.shape.sig:
                raise DataSetException("H5DataSet only supports whole-frame crops for now")
        data = np.ndarray(tileshape, dtype=dest_dtype)
        with self.reader.get_h5ds() as dataset:
            # FIXME: we currently transform back and forth between 3D and 4D
            # this should be possible to improve upon!
            slice_4d = unravel_nav(self.slice, self.meta.shape)
            subslices = list(slice_4d.subslices(shape=tileshape))
            for tile_slice in subslices:
                tile_slice_flat = tile_slice.flatten_nav(self.meta.shape)
                assert tile_slice_flat.shape.dims == 3
                assert tile_slice.shape.dims == 4
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
        # FIXME: this is not performance optimized, at all
        roi = roi.reshape((-1,))
        one_frame_shape = (1, 1) + tuple(self.meta.shape.sig)
        sig_origin = tuple([0] * self.meta.shape.sig.dims)
        frames_read = 0
        start_at_frame = self.slice.origin[0]
        frame_offset = np.count_nonzero(roi[:start_at_frame])
        for tile in self._get_tiles_normal(one_frame_shape, crop_to=None, dest_dtype=dest_dtype):
            if roi[tile.tile_slice.origin[0]]:
                tile_slice = Slice(
                    origin=(frames_read + frame_offset,) + sig_origin,
                    shape=tile.tile_slice.shape,
                )
                yield DataTile(
                    data=tile.data,
                    tile_slice=tile_slice,
                )
                frames_read += 1

    def get_tiles(self, crop_to=None, full_frames=False, mmap=False, dest_dtype="float32",
                  roi=None):
        if crop_to is not None and roi is not None:
            if crop_to.shape.nav.size != self._num_frames:
                raise ValueError("don't use crop_to with roi")
        if full_frames:
            tileshape = (
                tuple(self.tileshape.nav) + tuple(self.meta.shape.sig)
            )
        else:
            tileshape = self.tileshape
        if roi is not None:
            yield from self._get_tiles_with_roi(roi, dest_dtype)
        else:
            yield from self._get_tiles_normal(tileshape, crop_to, dest_dtype)
