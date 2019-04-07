import contextlib

import numpy as np
import h5py

from libertem.common import Slice, Shape
from .base import DataSet, Partition, DataTile, DataSetException, DataSetMeta


def _get_datasets(path):
    datasets = []

    def _make_list(name, obj):
        if hasattr(obj, 'size') and hasattr(obj, 'shape'):
            datasets.append((name, obj.size, obj.shape, obj.dtype))

    with h5py.File(path, 'r') as f:
        f.visititems(_make_list)
        for name, size, shape, dtype in sorted(datasets, key=lambda i: i[0]):
            yield {"name": name, "value": [
                {"name": "Size", "value": str(size)},
                {"name": "Shape", "value": str(shape)},
                {"name": "Datatype", "value": str(dtype)},
            ]}


class H5Reader(object):
    def __init__(self, path, ds_path):
        self._path = path
        self._ds_path = ds_path

    @contextlib.contextmanager
    def get_h5ds(self):
        with h5py.File(self._path, 'r') as f:
            yield f[self._ds_path]


class H5DataSet(DataSet):
    def __init__(self, path, ds_path, tileshape,
                 target_size=512*1024*1024, min_num_partitions=None, sig_dims=2):
        self.path = path
        self.ds_path = ds_path
        self.target_size = target_size
        self.sig_dims = sig_dims
        self.tileshape = Shape(tileshape, sig_dims=self.sig_dims)
        self.min_num_partitions = min_num_partitions
        self._dtype = None

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
                dtype=self._dtype,
            )
        return self

    @classmethod
    def detect_params(cls, path):
        try:
            with h5py.File(path, 'r'):
                pass
        except (IOError, OSError, KeyError, ValueError):
            # not a h5py file or can't open for some reason:
            return False

        # try to guess the hdf5 dataset path:
        datasets = []

        def _make_list(name, obj):
            if hasattr(obj, 'size') and hasattr(obj, 'shape'):
                datasets.append((name, obj.size, obj.shape, obj.dtype))
        with h5py.File(path, 'r') as f:
            f.visititems(_make_list)
        try:
            largest_ds = sorted(datasets, key=lambda i: i[1], reverse=True)[0]
            name, size, shape, dtype = largest_ds
        except IndexError:
            return {"path": path}

        return {
            "path": path,
            "ds_path": name,
            # FIXME: shape may not be 4D, number of frames may not match L3 size
            "tileshape": (1, 8) + shape[2:],
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
            return [
                {"name": "dtype", "value": str(ds.dtype)},
                {"name": "chunks", "value": str(ds.chunks)},
                {"name": "compression", "value": str(ds.compression)},
                {"name": "datasets", "value": list(_get_datasets(self.path))},
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
                partition_slice=pslice,
            )

    def __repr__(self):
        return "<H5DataSet of %s shape=%s>" % (self._dtype, self._shape)


class H5Partition(Partition):
    def __init__(self, tileshape, reader, *args, **kwargs):
        self.tileshape = tileshape
        self.reader = reader
        super().__init__(*args, **kwargs)

    def get_tiles(self, crop_to=None, full_frames=False):
        if crop_to is not None:
            if crop_to.shape.sig != self.meta.shape.sig:
                raise DataSetException("H5DataSet only supports whole-frame crops for now")
        if full_frames:
            tileshape = (
                tuple(self.tileshape[:self.meta.shape.nav.dims]) + tuple(self.meta.shape.sig)
            )
        else:
            tileshape = self.tileshape
        data = np.ndarray(tileshape, dtype=self.dtype)
        with self.reader.get_h5ds() as dataset:
            subslices = list(self.slice.subslices(shape=tileshape))
            for tile_slice in subslices:
                if crop_to is not None:
                    intersection = tile_slice.intersection_with(crop_to)
                    if intersection.is_null():
                        continue
                if tile_slice.shape != tileshape:
                    # at the border, can't reuse buffer
                    # hmm. aren't there only like 3 different shapes at the border?
                    # FIXME: use buffer pool to reuse buffers of same shape
                    border_data = np.ndarray(tile_slice.shape, dtype=self.dtype)
                    dataset.read_direct(border_data, source_sel=tile_slice.get())
                    yield DataTile(data=border_data, tile_slice=tile_slice)
                else:
                    # reuse buffer
                    dataset.read_direct(data, source_sel=tile_slice.get())
                    yield DataTile(data=data, tile_slice=tile_slice)
