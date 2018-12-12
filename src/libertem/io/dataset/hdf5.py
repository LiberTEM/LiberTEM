import contextlib

import numpy as np
import h5py

from libertem.common import Slice, Shape
from .base import DataSet, Partition, DataTile, DataSetException


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


class H5DataSet(DataSet):
    def __init__(self, path, ds_path, tileshape,
                 target_size=512*1024*1024, min_num_partitions=None):
        self.path = path
        self.ds_path = ds_path
        self.target_size = target_size
        self.sig_dims = 2  # FIXME!
        self.tileshape = Shape(tileshape, sig_dims=self.sig_dims)
        self.min_num_partitions = min_num_partitions

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

    @contextlib.contextmanager
    def get_h5ds(self):
        with h5py.File(self.path, 'r') as f:
            yield f[self.ds_path]

    @property
    def dtype(self):
        with self.get_h5ds() as h5ds:
            return h5ds.dtype

    @property
    def raw_shape(self):
        with self.get_h5ds() as h5ds:
            return Shape(h5ds.shape, sig_dims=self.sig_dims)

    def check_valid(self):
        try:
            with self.get_h5ds() as h5ds:
                h5ds.shape
            return True
        except (IOError, OSError, KeyError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def get_diagnostics(self):
        with self.get_h5ds() as ds:
            return [
                {"name": "dtype", "value": str(ds.dtype)},
                {"name": "chunks", "value": str(ds.chunks)},
                {"name": "compression", "value": str(ds.compression)},
                {"name": "datasets", "value": list(_get_datasets(self.path))},
            ]

    def get_partitions(self):
        with self.get_h5ds() as h5ds:
            ds_shape = Shape(h5ds.shape, sig_dims=self.sig_dims)
            ds_slice = Slice(origin=(0, 0, 0, 0), shape=ds_shape)
            partition_shape = self.partition_shape(
                datashape=h5ds.shape,
                framesize=h5ds[0][0].size,
                dtype=h5ds.dtype,
                target_size=self.target_size,
                min_num_partitions=self.min_num_partitions,
            )
            dtype = h5ds.dtype
            for pslice in ds_slice.subslices(partition_shape):
                # TODO: where should the tileshape be set? let the user choose for now
                yield H5Partition(
                    tileshape=self.tileshape,
                    dataset=self,
                    dtype=dtype,
                    partition_slice=pslice,
                )

    def __repr__(self):
        return "<H5DataSet of %s shape=%s>" % (self.dtype, self.shape)


class H5Partition(Partition):
    def __init__(self, tileshape, *args, **kwargs):
        self.tileshape = tileshape
        super().__init__(*args, **kwargs)

    def get_tiles(self, crop_to=None):
        if crop_to is not None:
            if crop_to.shape.sig != self.dataset.shape.sig:
                raise DataSetException("H5DataSet only supports whole-frame crops for now")
        data = np.ndarray(self.tileshape, dtype=self.dtype)
        with self.dataset.get_h5ds() as dataset:
            subslices = list(self.slice.subslices(shape=self.tileshape))
            for tile_slice in subslices:
                if crop_to is not None:
                    intersection = tile_slice.intersection_with(crop_to)
                    if intersection.is_null():
                        continue
                if tile_slice.shape != self.tileshape:
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
