import logging
from .base import Job, Task


log = logging.getLogger(__name__)


class PickFrameJob(Job):
    def __init__(self, slice_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slice = slice_

    def get_tasks(self):
        for partition in self.dataset.get_partitions():
            if self._slice.intersection_with(partition.slice).is_null():
                continue
            yield PickFrameTask(partition=partition, slice_=self._slice)


class PickFrameTask(Task):
    def __init__(self, slice_, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._slice = slice_

    def __call__(self):
        parts = []
        for data_tile in self.partition.get_tiles():
            intersection = data_tile.tile_slice.intersection_with(self._slice)
            if intersection.is_null():
                continue
            parts.append(PickFrameResultTile(data_tile=data_tile, intersection=intersection))
        return parts


class PickFrameResultTile(object):
    def __init__(self, data_tile, intersection):
        self.data_tile = data_tile
        self.intersection = intersection

    @property
    def dtype(self):
        return self.data_tile.dtype

    def _get_dest_slice(self):
        tile_slice = self.intersection.get()
        return (
            tile_slice[2],
            tile_slice[3],
        )

    def copy_to_result(self, result):
        dest_slice = self._get_dest_slice()
        shifted = self.intersection.shift(self.data_tile.tile_slice)
        result[dest_slice] = self.data_tile.data[shifted.get()]
        return result
