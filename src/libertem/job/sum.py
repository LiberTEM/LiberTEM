import numpy as np

from .base import Job, Task


class SumFramesJob(Job):
    def get_tasks(self):
        for partition in self.dataset.get_partitions():
            yield SumFramesTask(partition=partition)

    def get_result_shape(self):
        return self.dataset.shape[2:]


class SumFramesTask(Task):
    def __call__(self):
        """
        sum frames
        """
        part = np.zeros(self.partition.dataset.shape[2:], dtype="float32")
        for data_tile in self.partition.get_tiles():
            data = data_tile.data
            if data.dtype.kind == 'u':
                data = data.astype("float32")
            result = data_tile.data.sum(axis=(0, 1))
            tile_slice = data_tile.tile_slice.get()
            part[
                tile_slice[2],
                tile_slice[3],
            ] += result
        return [
            SumResultTile(
                data=part,
            )
        ]


class SumResultTile(object):
    def __init__(self, data):
        self.data = data

    @property
    def dtype(self):
        return self.data.dtype

    def copy_to_result(self, result):
        result += self.data
        return result
