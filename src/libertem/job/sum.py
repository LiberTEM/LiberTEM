from .base import Job, Task


class SumFramesJob(Job):
    def get_tasks(self):
        for partition in self.dataset.get_partitions():
            yield SumFramesTask(partition=partition)


class SumFramesTask(Task):
    def __call__(self):
        """
        sum frames
        """
        parts = []
        for data_tile in self.partition.get_tiles():
            data = data_tile.data
            if data.dtype.kind == 'u':
                data = data.astype("float32")
            result = data_tile.data.sum(axis=(0, 1))
            parts.append(
                SumResultTile(
                    data=result,
                    tile_slice=data_tile.tile_slice,
                )
            )
        return parts


class SumResultTile(object):
    def __init__(self, data, tile_slice):
        self.data = data
        self.tile_slice = tile_slice

    @property
    def dtype(self):
        return self.data.dtype

    def _get_dest_slice(self):
        tile_slice = self.tile_slice.get()
        return (
            tile_slice[2],
            tile_slice[3],
        )

    def copy_to_result(self, result):
        # FIXME: assumes tile size is less than or equal one row of frames. is this true?
        # let's assert it for now:
        assert self.tile_slice.shape[0] == 1

        result[self._get_dest_slice()] += self.data
        return result
