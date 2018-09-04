import contextlib

import numpy as np
import h5py

from libertem.common.slice import Slice
from .base import DataSet, Partition, DataTile, DataSetException


class H5DataSet(DataSet):
    def __init__(self, path, ds_path, tileshape,
                 target_size=512*1024*1024, min_num_partitions=None):
        self.path = path
        self.ds_path = ds_path
        self.target_size = target_size
        self.tileshape = tileshape
        self.min_num_partitions = min_num_partitions

    @contextlib.contextmanager
    def get_h5ds(self):
        with h5py.File(self.path, 'r') as f:
            yield f[self.ds_path]

    @property
    def dtype(self):
        with self.get_h5ds() as h5ds:
            return h5ds.dtype

    @property
    def shape(self):
        with self.get_h5ds() as h5ds:
            return h5ds.shape

    def check_valid(self):
        try:
            with self.get_h5ds() as h5ds:
                h5ds.shape
            return True
        except (IOError, OSError, KeyError, ValueError) as e:
            raise DataSetException("invalid dataset: %s" % e)

    def _get_datasets(self):
        datasets = []

        def _make_list(name, obj):
            if hasattr(obj, 'size') and hasattr(obj, 'shape'):
                datasets.append((name, obj.size, obj.shape, obj.dtype))

        with h5py.File(self.path, 'r') as f:
            f.visititems(_make_list)
            for name, size, shape, dtype in sorted(datasets, key=lambda i: i[0]):
                yield {"name": name, "value": [
                    {"name": "Size", "value": str(size)},
                    {"name": "Shape", "value": str(shape)},
                    {"name": "Datatype", "value": str(dtype)},
                ]}

    def get_diagnostics(self):
        with self.get_h5ds() as ds:
            return [
                {"name": "dtype", "value": str(ds.dtype)},
                {"name": "chunks", "value": str(ds.chunks)},
                {"name": "compression", "value": str(ds.compression)},
                {"name": "datasets", "value": list(self._get_datasets())},
            ]

    def get_partitions(self):
        with self.get_h5ds() as h5ds:
            ds_slice = Slice(origin=(0, 0, 0, 0), shape=h5ds.shape)
            partition_shape = Slice.partition_shape(
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
            if crop_to.shape[2:] != self.dataset.shape[2:]:
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

    def get_locations(self):
        return "127.0.1.1"  # FIXME
