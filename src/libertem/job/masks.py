import functools
import logging

try:
    import torch
except ImportError:
    torch = None
import scipy.sparse as sp
import numpy as np

from libertem.io.dataset.base import DataTile, Partition
from .base import Job, Task, ResultTile
from libertem.masks import to_dense, to_sparse
from libertem.common import Slice

log = logging.getLogger(__name__)


def _make_mask_slicer(computed_masks):
    @functools.lru_cache(maxsize=None)
    def _get_masks_for_slice(slice_):
        sliced_masks = [
            # .reshape((-1, 1)) -> like flatten, but compatible with sparse
            # matrices and no copies
            # should save us one copy as we np.hstack() immediately afterwards
            # https://stackoverflow.com/a/28930580/540644
            slice_.get(mask, sig_only=True).reshape((-1, 1))
            for mask in computed_masks
        ]
        # MaskContainer assures that all or none of the masks are sparse
        if sp.issparse(sliced_masks[0]):
            return sp.hstack(sliced_masks)
        else:
            return np.hstack(sliced_masks)
    return _get_masks_for_slice


class ApplyMasksJob(Job):
    """
    Apply masks to signals/frames in the dataset.
    """
    def __init__(self, mask_factories, use_torch=True, use_sparse=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        mask_dtype = np.dtype(self.dataset.dtype)
        if mask_dtype.kind in ('u', 'i'):
            mask_dtype = np.dtype("float32")
        self.masks = MaskContainer(mask_factories, dtype=mask_dtype, use_sparse=use_sparse)
        self.use_torch = use_torch

    def get_tasks(self):
        for idx, partition in enumerate(self.dataset.get_partitions()):
            yield ApplyMasksTask(
                partition=partition,
                masks=self.masks,
                use_torch=self.use_torch,
                idx=idx,
            )

    def get_result_shape(self):
        return (len(self.masks),) + tuple(self.dataset.raw_shape.nav)


class MaskContainer(object):
    def __init__(self, mask_factories, dtype, use_sparse=None):
        self.mask_factories = mask_factories
        self.dtype = dtype
        self.use_sparse = use_sparse
        self._mask_cache = {}
        # lazily initialized in the worker process, to keep task size small:
        self._computed_masks = None
        self._get_masks_for_slice = None
        self.validate_mask_functions()

    def validate_mask_functions(self):
        for fn in self.mask_factories:
            try:
                if 'self' in fn.__code__.co_freevars:
                    log.warn('mask factory closes over self, may be inefficient')
            except Exception:
                raise

    def __len__(self):
        return len(self.mask_factories)

    def __getitem__(self, key):
        if isinstance(key, Partition):
            slice_ = key.slice
        elif isinstance(key, DataTile):
            slice_ = key.tile_slice
        elif isinstance(key, Slice):
            slice_ = key
        else:
            raise TypeError(
                "MaskContainer[k] can only be called with "
                "DataTile/Slice/Partition instances"
            )
        return self.get_masks_for_slice(slice_.discard_nav())

    @property
    def shape(self):
        m0 = self.computed_masks[0]
        return (m0.size, len(self.computed_masks))

    def _compute_masks(self):
        """
        Call mask factories and convert to the dataset dtype

        Returns
        -------
        a list of masks with contents as they were created by the factories
        and converted uniformly to dense or sparse matrices depending on
        ``self.use_sparse``.
        """
        # Make sure all the masks are either sparse or dense
        # If the use_sparse property is set to Ture or False,
        # it takes precedence.
        # If it is None, use sparse only if all masks are sparse
        # and set the use_sparse property accordingly

        raw_masks = [
            f().astype(self.dtype)
            for f in self.mask_factories
        ]
        if self.use_sparse is True:
            masks = [
                to_sparse(m) for m in raw_masks
            ]
        elif self.use_sparse is False:
            masks = [
                to_dense(m) for m in raw_masks
            ]
        else:
            sparse = [
                sp.issparse(m) for m in raw_masks
            ]
            if all(sparse):
                self.use_sparse = True
                masks = raw_masks
            else:
                self.use_sparse = False
                masks = [
                    to_dense(m) for m in raw_masks
                ]
        return masks

    def get_masks_for_slice(self, slice_):
        if self._get_masks_for_slice is None:
            self._get_masks_for_slice = _make_mask_slicer(self.computed_masks)
        return self._get_masks_for_slice(slice_)

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
        super().__init__(*args, **kwargs)
        self.masks = masks
        self.use_torch = use_torch
        if torch is None or np.dtype(self.partition.dtype).kind == 'c':
            self.use_torch = False

    def reshaped_data(self, data, dest_slice):
        """
        Reshapes the result from the flattened and interleaved version to a shape
        that fits the result array (masks, ...nav_dims)
        """

        num_masks = data.shape[1]

        deinterleaved = np.stack(
            [data.ravel()[idx::num_masks]
             for idx in range(num_masks)],
            axis=0,
        )
        return deinterleaved.reshape((num_masks,) + tuple(dest_slice.shape.nav))

    def __call__(self):
        num_masks = len(self.masks)
        dest_dtype = np.dtype(self.partition.dtype)
        if dest_dtype.kind not in ('c', 'f'):
            dest_dtype = 'float32'
        part = np.zeros((num_masks,) + tuple(self.partition.shape.nav), dtype=dest_dtype)
        for data_tile in self.partition.get_tiles():
            flat_data = data_tile.flat_data
            if flat_data.dtype != dest_dtype:
                data = flat_data.astype(dest_dtype)
            else:
                data = flat_data
            masks = self.masks[data_tile]
            if self.masks.use_sparse:
                # The sparse matrix has to be the left-hand side, for that
                # reason we transpose before and after multiplication.
                result = masks.T.dot(data.T).T
            elif self.use_torch:
                result = torch.mm(
                    torch.from_numpy(data),
                    torch.from_numpy(masks),
                ).numpy()
            else:
                result = data.dot(masks)
            dest_slice = data_tile.tile_slice.shift(self.partition.slice)
            reshaped = self.reshaped_data(data=result, dest_slice=dest_slice)
            # Ellipsis to match the "number of masks" part of the result
            part[(Ellipsis,) + dest_slice.get(nav_only=True)] += reshaped
        return [
            MaskResultTile(
                data=part,
                dest_slice=self.partition.slice.get(nav_only=True),
            )
        ]


class MaskResultTile(ResultTile):
    def __init__(self, data, dest_slice):
        self.data = data
        self.dest_slice = dest_slice

    def __repr__(self):
        return "<ResultTile for slice=%r>" % self.dest_slice

    @property
    def dtype(self):
        return self.data.dtype

    def reduce_into_result(self, result):
        result[(Ellipsis,) + self.dest_slice] += self.data
        return result
