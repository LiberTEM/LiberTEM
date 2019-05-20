import functools
import logging
from collections.abc import Iterable

try:
    import torch
except ImportError:
    torch = None
import sparse
import scipy.sparse
import numpy as np

from libertem.io.dataset.base import DataTile, Partition
from .base import Job, Task, ResultTile
from libertem.masks import to_dense, to_sparse, is_sparse
from libertem.common import Slice
from libertem.common.buffers import zeros_aligned

log = logging.getLogger(__name__)


def _make_mask_slicer(computed_masks):
    @functools.lru_cache(maxsize=None)
    def _get_masks_for_slice(slice_):
        stack_height = computed_masks.shape[0]
        m = slice_.get(computed_masks, sig_only=True)
        # We need the mask's signal dimension flattened and the stack transposed in the next step
        # For that reason we flatten and transpose here so that we use the cache of this function.
        return m.reshape((stack_height, -1)).T
    return _get_masks_for_slice


class ApplyMasksJob(Job):
    """
    Apply masks to signals/frames in the dataset.
    """
    def __init__(self, mask_factories, use_torch=True, use_sparse=None, length=None,
                *args, **kwargs):
        super().__init__(*args, **kwargs)
        mask_dtype = np.dtype(self.dataset.dtype)
        if mask_dtype.kind in ('u', 'i'):
            mask_dtype = np.dtype("float32")
        self.masks = MaskContainer(mask_factories, dtype=mask_dtype,
            use_sparse=use_sparse, length=length)
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
        return (len(self.masks),) + tuple(self.dataset.shape.flatten_nav().nav)


class MaskContainer(object):
    def __init__(self, mask_factories, dtype, use_sparse=None, length=None):
        self.mask_factories = mask_factories
        # If we generate a whole mask stack with one function call,
        # we should know the length without generating the mask stack
        self.length = length
        if callable(mask_factories) and length is None:
            raise TypeError("The length parameter has to be set if mask_factories is not iterable.")
        self.dtype = dtype
        self.use_sparse = use_sparse
        self._mask_cache = {}
        # lazily initialized in the worker process, to keep task size small:
        self._computed_masks = None
        self._get_masks_for_slice = None
        self.validate_mask_functions()

    def validate_mask_functions(self):
        fns = self.mask_factories
        if not isinstance(fns, Iterable):
            fns = [fns]
        for fn in fns:
            try:
                if 'self' in fn.__code__.co_freevars:
                    log.warning('mask factory closes over self, may be inefficient')
            except Exception:
                raise

    def __len__(self):
        if isinstance(self.mask_factories, Iterable):
            return len(self.mask_factories)
        else:
            return self.length

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

        if callable(self.mask_factories):
            raw_masks = self.mask_factories().astype(self.dtype)
            default_sparse = is_sparse(raw_masks)
            mask_slices = [raw_masks]
        else:
            mask_slices = []
            default_sparse = True
            for f in self.mask_factories:
                m = f().astype(self.dtype)
                # Scipy.sparse is always 2D, so we have to convert here
                # before reshaping
                if scipy.sparse.issparse(m):
                    m = sparse.COO.from_scipy_sparse(m)
                # We reshape to be a stack of 1 so that we can unify code below
                m = m.reshape((1, ) + m.shape)
                default_sparse = default_sparse and is_sparse(m)
                mask_slices.append(m)

        if self.use_sparse is None:
            self.use_sparse = default_sparse

        if self.use_sparse:
            masks = sparse.concatenate(
                [to_sparse(m) for m in mask_slices]
            )
        else:
            masks = np.concatenate(
                [to_dense(m) for m in mask_slices]
            )
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
        part = zeros_aligned((num_masks,) + tuple(self.partition.shape.nav), dtype=dest_dtype)
        for data_tile in self.partition.get_tiles(mmap=True, dest_dtype=dest_dtype):
            flat_data = data_tile.flat_data
            masks = self.masks[data_tile]
            if self.masks.use_sparse:
                result = sparse.dot(flat_data, masks)
            elif self.use_torch:
                result = torch.mm(
                    torch.from_numpy(flat_data),
                    torch.from_numpy(masks),
                ).numpy()
            else:
                result = flat_data.dot(masks)
            dest_slice = data_tile.tile_slice.shift(self.partition.slice)
            reshaped = self.reshaped_data(data=result, dest_slice=dest_slice)
            # Ellipsis to match the "number of masks" part of the result
            part[(...,) + dest_slice.get(nav_only=True)] += reshaped
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
