import operator
import functools
import contextlib

import numpy as np
import h5py

from .base import DataSet, Partition
from ..slice import Slice
from ..tiling import DataTile


class H5DataSet(DataSet):
    def __init__(self, path, ds_path, stackheight=8, target_size=512*1024*1024):
        self.path = path
        self.ds_path = ds_path
        self.target_size = target_size
        self.stackheight = stackheight

    @contextlib.contextmanager
    def get_h5ds(self):
        with h5py.File(self.path, 'r') as f:
            yield f[self.ds_path]

    @property
    def dtype(self):
        with self.get_h5ds() as h5ds:
            return h5ds.dtype

    def get_partitions(self):
        with self.get_h5ds() as h5ds:
            ds_slice = Slice(origin=(0, 0, 0, 0), shape=h5ds.shape)
            partition_shape = Slice.partition_shape(
                datashape=h5ds.shape,
                framesize=h5ds[0][0].size,
                dtype=h5ds.dtype,
                target_size=self.target_size
            )
            framesize = functools.reduce(operator.mul, tuple(h5ds.shape[-2:]))
            dtype = h5ds.dtype
            for pslice in ds_slice.subslices(partition_shape):
                yield H5Partition(
                    tileshape=(self.stackheight, framesize),
                    dataset=self,
                    dtype=dtype,
                    partition_slice=pslice,
                )


class H5Partition(Partition):
    def __init__(self, tileshape, *args, **kwargs):
        self.tileshape = tileshape
        super().__init__(*args, **kwargs)

    def get_tiles(self):
        data = np.ndarray(self.tileshape, dtype=self.dtype)
        data_subslice_view = data.reshape(
            (1, self.tileshape[0],
             self.slice.shape[2],
             self.slice.shape[3])
        )
        with self.dataset.get_h5ds() as dataset:
            assert (self.slice.shape[0] * self.slice.shape[1]) % self.tileshape[0] == 0,\
                "please chose a tileshape that evenly divides the partition"
            # num_stacks is only computed for comparison to subslices
            num_stacks = (self.slice.shape[0] * self.slice.shape[1]) // self.tileshape[0]
            # NOTE: computation is done on (stackheight, framesize) tiles, but logically, they
            # are equivalent to tiles of shape (1, stackheight, frameheight, framewidth)
            subslices = list(self.slice.subslices(shape=(1, self.tileshape[0],
                                                         self.slice.shape[2],
                                                         self.slice.shape[3])))
            assert num_stacks == len(subslices)
            for tile_slice in subslices:
                dataset.read_direct(data_subslice_view, source_sel=tile_slice.get())
                yield DataTile(data=data, tile_slice=tile_slice)

    def get_locations(self):
        return "127.0.1.1"  # FIXME
