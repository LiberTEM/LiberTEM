import numpy as np

from .base import Job, Task
from ..tiling import ResultTile


class ApplyMasksJob(Job):
    def __init__(self, masks, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.orig_masks = masks
        self.masks = self._merge_masks(masks)

    def _merge_masks(self, masks):
        """
        flatten, convert and merge masks into one array

        Parameters
        ----------
        masks : [ndarray]
            list of 2D arrays that represent masks
        """
        masks = [m.flatten().astype(self.dataset.dtype) for m in masks]
        return np.stack(masks, axis=1)

    def get_tasks(self):
        for partition in self.dataset.get_partitions():
            yield ApplyMasksTask(partition=partition, masks=self.masks)

    @property
    def maskcount(self):
        return len(self.orig_masks)


class ApplyMasksTask(Task):
    def __init__(self, masks, *args, **kwargs):
        self.masks = masks
        super().__init__(*args, **kwargs)

    def __call__(self):
        parts = []
        for data_tile in self.partition.get_tiles():
            result = data_tile.data.dot(self.masks)
            parts.append(
                ResultTile(
                    data=result,
                    tile_slice=data_tile.tile_slice,
                )
            )
        return parts
