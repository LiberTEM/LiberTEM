import logging

from libertem.udf.base import Task
from .base import BaseJob, ResultTile
from libertem.common.buffers import zeros_aligned


log = logging.getLogger(__name__)


class PickFrameJob(BaseJob):
    '''
    .. deprecated:: 0.4.0
        Use :meth:`libertem.api.Context.create_pick_analysis`, :class:`libertem.udf.raw.PickUDF`,
        :class:`libertem.udf.masks.ApplyMasksUDF`or a custom UDF (:ref:`user-defined functions`)
        as a replacement. See also :ref:`job deprecation`.
    '''
    def __init__(self, slice_, squeeze=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slice = slice_
        self._squeeze = squeeze
        assert slice_.shape.nav.dims == 1, "slice must have flat nav"

    def get_tasks(self):
        for idx, partition in enumerate(self.dataset.get_partitions()):
            if self._slice.intersection_with(partition.slice).is_null():
                continue
            yield PickFrameTask(partition=partition, slice_=self._slice, idx=idx)

    def get_result_shape(self):
        if self._squeeze:
            return tuple(part for part in self._slice.shape
                         if part > 1)
        return self._slice.shape


class PickFrameTask(Task):
    def __init__(self, slice_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slice = slice_

    def __call__(self):
        result = zeros_aligned(self._slice.shape, dtype=self.partition.dtype)
        for data_tile in self.partition.get_tiles(crop_to=self._slice, mmap=True):
            intersection = data_tile.tile_slice.intersection_with(self._slice)
            # shift to data_tile relative coordinates:
            shifted = intersection.shift(data_tile.tile_slice)
            result[
                intersection.shift(self._slice).get()
            ] = data_tile.data[shifted.get()]
        return [PickFrameResultTile(data=result)]


class PickFrameResultTile(ResultTile):
    def __init__(self, data):
        self.data = data

    @property
    def dtype(self):
        return self.data.dtype

    def reduce_into_result(self, result):
        out = result.reshape(self.data.shape)
        out += self.data
        return result
