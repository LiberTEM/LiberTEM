import functools

try:
    import torch
except ImportError:
    torch = None
import numpy as np

from libertem.io.dataset.base import DataTile, Partition
from .base import Job, Task


def _make_mask_slicer(computed_masks):
    @functools.lru_cache(maxsize=None)
    def _get_masks_for_slice(slice_):
        """
        """
        sliced_masks = [
            # .reshape((-1,)) -> like flatten, but no copies
            # should save us one copy as we np.stack() immediately afterwards
            # https://stackoverflow.com/a/28930580/540644
            slice_.get(mask, signal_only=True).reshape((-1,))
            for mask in computed_masks
        ]
        return np.stack(sliced_masks, axis=1)
    return _get_masks_for_slice


class ApplyMasksJob(Job):
    def __init__(self, mask_factories, use_torch=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mask_dtype = np.dtype(self.dataset.dtype)
        if mask_dtype.kind == 'u':
            mask_dtype = np.dtype("float32")
        self.masks = MaskContainer(mask_factories, dtype=mask_dtype)
        self.use_torch = use_torch

    def get_tasks(self):
        for partition in self.dataset.get_partitions():
            yield ApplyMasksTask(
                partition=partition,
                masks=self.masks,
                use_torch=self.use_torch,
            )

    def get_result_shape(self):
        return (len(self.masks),) + self.dataset.shape[:2]


class MaskContainer(object):
    def __init__(self, mask_factories, dtype):
        self.mask_factories = mask_factories
        self.dtype = dtype
        self._mask_cache = {}
        # lazily initialized in the worker process, to keep task size small:
        self._computed_masks = None
        self._get_masks_for_slice = None

    def __len__(self):
        return len(self.mask_factories)

    def __getitem__(self, key):
        if isinstance(key, Partition):
            slice_ = key.slice
        elif isinstance(key, (DataTile, ResultTile)):
            slice_ = key.tile_slice
        else:
            raise TypeError(
                "MaskContainer[k] can only be called with DataTile/ResultTile instances"
            )
        return self.get_masks_for_slice(slice_)

    @property
    def shape(self):
        m0 = self.computed_masks[0]
        return (m0.size, len(self.computed_masks))

    def _compute_masks(self):
        """
        Call mask factories and convert to the dataset dtype

        Returns
        -------
        a list of masks as they were created by the factories
        """
        return [f().astype(self.dtype)
                for f in self.mask_factories]

    def get_masks_for_slice(self, slice_):
        if self._get_masks_for_slice is None:
            self._get_masks_for_slice = _make_mask_slicer(self.computed_masks)
        return self._get_masks_for_slice(slice_.discard_scan())

    @property
    def computed_masks(self):
        if self._computed_masks is None:
            self._computed_masks = self._compute_masks()
        return self._computed_masks


class ApplyMasksTask(Task):
    def __init__(self, masks, use_torch, *args, **kwargs):
        """
        Parameters
        ----------
        partition : libertem.dataset.base.Partition instance
            the partition to work on
        masks : MaskContainer
            the masks to apply to the partition
        """
        self.masks = masks
        self.use_torch = use_torch
        super().__init__(*args, **kwargs)

    def __call__(self):
        parts = []
        for data_tile in self.partition.get_tiles():
            # print("dotting\n%r\nwith\n%r\n\n" % (data_tile.flat_data, self.masks[data_tile]))
            data = data_tile.flat_data
            if data.dtype.kind == 'u':
                data = data.astype("float32")
            masks = self.masks[data_tile]
            if self.use_torch and torch is not None:
                result = torch.mm(
                    torch.from_numpy(data),
                    torch.from_numpy(masks),
                ).numpy()
            else:
                result = data.dot(masks)
            parts.append(
                ResultTile(
                    data=result,
                    tile_slice=data_tile.tile_slice,
                )
            )
        return parts


class ResultTile(object):
    def __init__(self, data, tile_slice):
        self.data = data
        self.tile_slice = tile_slice

    def __repr__(self):
        return "<ResultTile for slice=%r>" % self.tile_slice

    def _get_dest_slice_3d(self, result_shape):
        tile_slice = self.tile_slice
        sy0 = tile_slice.origin[0]
        sx0 = tile_slice.origin[1]

        start_idx = sy0 * result_shape[2] + sx0
        num_frames = tile_slice.shape[0] * tile_slice.shape[1]

        return (
            Ellipsis,
            slice(start_idx, start_idx + num_frames),
        )

    def _get_dest_slice_4d(self):
        tile_slice = self.tile_slice.get()

        return (
            Ellipsis,
            tile_slice[0],
            tile_slice[1],
        )

    def reshaped_data(self, flat_frames=True):
        """
        Reshapes the result from the flattened and interleaved version to a shape
        that fits the result array (masks, y, x) or (masks, num_frames)
        """

        num_masks = self.data.shape[1]

        deinterleaved = np.stack(
            [self.data.ravel()[idx::num_masks]
             for idx in range(num_masks)],
            axis=0,
        )
        if flat_frames:
            return deinterleaved.reshape(
                num_masks,
                self.tile_slice.shape[0] * self.tile_slice.shape[1],
            )
        else:
            return deinterleaved.reshape(
                num_masks,
                self.tile_slice.shape[0],
                self.tile_slice.shape[1],
            )

    @property
    def dtype(self):
        return self.data.dtype

    def copy_to_result(self, result):
        # is this tile contiguous on (num_masks, num_frames)?
        if self.tile_slice.shape[0] == 1 or self.tile_slice.shape[1] == result.shape[2]:
            s = result.shape
            result_flat = result.reshape((s[0], s[1] * s[2]))
            result_flat[self._get_dest_slice_3d(s)] += self.reshaped_data()
        else:
            result[self._get_dest_slice_4d()] += self.reshaped_data(flat_frames=False)
        return result
