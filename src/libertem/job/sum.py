import numpy as np

from .base import Job, Task, ResultTile
from libertem.common.buffers import zeros_aligned


class SumFramesJob(Job):
    def get_tasks(self):
        for idx, partition in enumerate(self.dataset.get_partitions()):
            yield SumFramesTask(partition=partition, idx=idx)

    def get_result_shape(self):
        return self.dataset.shape.sig


class SumFramesTask(Task):
    def __call__(self):
        """
        sum frames over navigation axes
        """
        dest_dtype = np.dtype(self.partition.dtype)
        if dest_dtype.kind not in ('c', 'f'):
            dest_dtype = 'float32'
        part = zeros_aligned(self.partition.meta.shape.sig, dtype=dest_dtype)
        buf = zeros_aligned(self.partition.meta.shape.sig, dtype=dest_dtype)
        for data_tile in self.partition.get_tiles(dest_dtype=dest_dtype, mmap=True):
            data_tile.data.sum(axis=0, out=buf)
            part[data_tile.tile_slice.get(sig_only=True)] += buf

        return [
            SumResultTile(
                data=part,
            )
        ]


class SumResultTile(ResultTile):
    def __init__(self, data):
        self.data = data

    @property
    def dtype(self):
        return self.data.dtype

    def reduce_into_result(self, result):
        result += self.data
        return result
