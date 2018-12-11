import numpy as np

from libertem.io.dataset.base import DataTile, DataSet, Partition
from libertem.common import Slice, Shape
from libertem.masks import to_dense


class MemoryDataSet(DataSet):
    def __init__(self, data, tileshape, partition_shape, sig_dims=2, effective_shape=None):
        self.data = data
        self.tileshape = Shape(tileshape, sig_dims=sig_dims)
        self.partition_shape = Shape(partition_shape, sig_dims=sig_dims)
        self.sig_dims = sig_dims
        self._effective_shape = effective_shape and Shape(effective_shape, sig_dims) or None

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return Shape(self.data.shape, sig_dims=self.sig_dims)

    @property
    def effective_shape(self):
        return self._effective_shape or self.shape

    def check_valid(self):
        return True

    def get_partitions(self):
        ds_slice = Slice(origin=tuple([0] * self.shape.dims), shape=self.shape)
        for pslice in ds_slice.subslices(self.partition_shape):
            yield MemoryPartition(
                tileshape=self.tileshape,
                dataset=self,
                dtype=self.dtype,
                partition_slice=pslice,
            )


class MemoryPartition(Partition):
    def __init__(self, tileshape, *args, **kwargs):
        self.tileshape = tileshape
        super().__init__(*args, **kwargs)

    def get_tiles(self, crop_to=None):
        subslices = self.slice.subslices(shape=self.tileshape)
        for tile_slice in subslices:
            if crop_to is not None:
                intersection = tile_slice.intersection_with(crop_to)
                if intersection.is_null():
                    continue
            yield DataTile(
                data=self.dataset.data[tile_slice.get()],
                tile_slice=tile_slice
            )

    def __repr__(self):
        return "<MemoryPartition for %r>" % self.slice


def _naive_mask_apply(masks, data):
    """
    masks: list of masks
    data: 4d array of input data

    returns array of shape (num_masks, scan_y, scan_x)
    """
    assert len(data.shape) == 4
    for mask in masks:
        assert mask.shape == data.shape[2:], "mask doesn't fit frame size"

    res = np.zeros((len(masks),) + tuple(data.shape[:2]))
    for n in range(len(masks)):
        mask = to_dense(masks[n])
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                item = data[i, j].ravel().dot(mask.ravel())
                res[n, i, j] = item
    return res
