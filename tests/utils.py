import numpy as np

from libertem.io.dataset.base import DataTile, DataSet, Partition
from libertem.common.slice import Slice


class MemoryDataSet(DataSet):
    def __init__(self, data, tileshape, partition_shape):
        self.data = data
        self.tileshape = tileshape
        self.partition_shape = partition_shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def shape(self):
        return self.data.shape

    def check_valid(self):
        return True

    def get_partitions(self):
        ds_slice = Slice(origin=(0, 0, 0, 0), shape=self.shape)
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
        subslices = list(self.slice.subslices(shape=self.tileshape))
        for tile_slice in subslices:
            if crop_to is not None:
                intersection = tile_slice.intersection_with(crop_to)
                if intersection.is_null():
                    continue
            yield DataTile(
                data=self.dataset.data[tile_slice.get()],
                tile_slice=tile_slice
            )

    def get_locations(self):
        return "127.0.1.1"  # FIXME

    def __repr__(self):
        return "<MemoryPartition for %r>" % self.slice


def _naive_mask_apply(masks, data):
    """
    masks: list of masks
    data: 4d array of input data

    returns array of shape (num_masks, scan_y, scan_x)
    """
    res = np.zeros((len(masks),) + tuple(masks[0].shape))
    for n in range(len(masks)):
        mask = masks[n]
        for i in range(mask.shape[0]):
            for j in range(mask.shape[1]):
                item = data[i, j].ravel().dot(mask.ravel())
                res[n, i, j] = item
    return res
