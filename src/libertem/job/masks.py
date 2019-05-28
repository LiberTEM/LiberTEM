import functools
import logging

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


def _make_mask_slicer(computed_masks, dtype):
    @functools.lru_cache(maxsize=None)
    def _get_masks_for_slice(slice_):
        stack_height = computed_masks.shape[0]
        m = slice_.get(computed_masks, sig_only=True)
        # We need the mask's signal dimension flattened and the stack transposed in the next step
        # For that reason we flatten and transpose here so that we use the cache of this function.
        m = m.reshape((stack_height, -1)).T
        if is_sparse(m):
            iis, jjs = m.coords
            values = m.data
            # Just for calculation: scipy.sparse.csr_matrix is
            # the fastest for dot product
            return scipy.sparse.csr_matrix((values, (iis, jjs)), shape=m.shape, dtype=dtype)
        else:
            # We convert to the desired type.
            # This makes sure it is in row major, dense layout as well
            return m.astype(dtype)
    return _get_masks_for_slice


class ApplyMasksJob(Job):
    """
    Apply masks to signals/frames in the dataset.
    """
    def __init__(self, mask_factories, use_torch=True, use_sparse=None, mask_count=None,
                mask_dtype=None, dtype=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masks = MaskContainer(mask_factories, dtype=mask_dtype,
            use_sparse=use_sparse, count=mask_count)

        self.dtype = dtype
        self.use_torch = use_torch

    def get_tasks(self):
        for idx, partition in enumerate(self.dataset.get_partitions()):
            yield ApplyMasksTask(
                partition=partition,
                masks=self.masks,
                use_torch=self.use_torch,
                idx=idx,
                dtype=self.get_result_dtype()
            )

    def get_result_shape(self):
        return (len(self.masks),) + tuple(self.dataset.shape.flatten_nav().nav)

    def get_result_dtype(self):
        def is_wide(dtype):
            dtype = np.dtype(dtype)
            result = False
            if dtype.kind != 'c' and dtype.itemsize > 4:
                result = True
            if dtype.kind == 'c' and dtype.itemsize > 8:
                result = True
            return result

        if self.dtype is None:
            default_dtype = np.float32
            if is_wide(self.dataset.dtype) or is_wide(self.masks.dtype):
                default_dtype = np.float64
            return np.result_type(default_dtype, self.dataset.dtype, self.masks.dtype)
        else:
            return self.dtype


class MaskContainer(object):
    def __init__(self, mask_factories, dtype=None, use_sparse=None, count=None):
        self.mask_factories = mask_factories
        # If we generate a whole mask stack with one function call,
        # we should know the length without generating the mask stack
        self._length = count
        self._dtype = dtype
        self.use_sparse = use_sparse
        self._mask_cache = {}
        # lazily initialized in the worker process, to keep task size small:
        self._computed_masks = None
        self._get_masks_for_slice = {}
        self.validate_mask_functions()

    def validate_mask_functions(self):
        fns = self.mask_factories
        if callable(fns):
            fns = [fns]
        for fn in fns:
            try:
                # functools.partial doesn't have a __code__ attribute
                # use fn.func.c__code__
                # FIXME check the args and kwargs of fn for inefficiencies?
                if isinstance(fn, functools.partial):
                    if 'self' in fn.func.__code__.co_freevars:
                        log.warning('mask factory closes over self, may be inefficient')
                else:

                    if 'self' in fn.__code__.co_freevars:
                        log.warning('mask factory closes over self, may be inefficient')
            except Exception:
                raise

    def __len__(self):
        if self._length is not None:
            return self._length
        elif not callable(self.mask_factories):
            return len(self.mask_factories)
        else:
            return len(self.computed_masks)

    def get(self, key, dtype=None):
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
        return self.get_masks_for_slice(slice_.discard_nav(), dtype=dtype)

    @property
    def dtype(self):
        if self._dtype is None:
            return self.computed_masks.dtype
        else:
            return self._dtype

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
            raw_masks = self.mask_factories()
            default_sparse = is_sparse(raw_masks)
            mask_slices = [raw_masks]
        else:
            mask_slices = []
            default_sparse = True
            for f in self.mask_factories:
                m = f()
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

    def get_masks_for_slice(self, slice_, dtype=None):
        if dtype is None:
            dtype = self.dtype
        if dtype not in self._get_masks_for_slice:
            self._get_masks_for_slice[dtype] = _make_mask_slicer(self.computed_masks, dtype=dtype)
        return self._get_masks_for_slice[dtype](slice_)

    @property
    def computed_masks(self):
        if self._computed_masks is None:
            self._computed_masks = self._compute_masks()
        return self._computed_masks


class ApplyMasksTask(Task):
    def __init__(self, masks, use_torch, dtype, *args, **kwargs):
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
        self.dtype = np.dtype(dtype)
        # FIXME check the performance impact of mixing 32 bit and 64 bit,
        # real and complex types in dot product. It might be advantageous to
        # adjust the read and mask dtypes here so that the operations in __call__() use
        # the most convenient dtype for mask and dataset.
        self.read_dtype = self._input_dtype(self.partition.dtype)
        self.mask_dtype = self._input_dtype(self.masks.dtype)
        if torch is None or self.dtype.kind != 'f' or self.read_dtype != self.mask_dtype:
            self.use_torch = False

    def _input_dtype(self, dtype):
        dtype = np.dtype(dtype)
        # Convert integer data to floats if we want to produce floats or complex
        if dtype.kind in ('u', 'i', 'b') and self.dtype.kind in ('f', 'c'):
            # We have int64 or similar, use float64 to fit as much information as possible.
            if dtype.itemsize > 4:
                dtype = np.float64
            else:
                dtype = np.float32
        return dtype

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
        part = zeros_aligned((num_masks,) + tuple(self.partition.shape.nav), dtype=self.dtype)
        for data_tile in self.partition.get_tiles(mmap=True, dest_dtype=self.read_dtype):
            flat_data = data_tile.flat_data
            masks = self.masks.get(data_tile, self.mask_dtype)
            if self.masks.use_sparse:
                # This is scipy.sparse using the old matrix interface
                # where "*" is the dot product
                result = flat_data * masks
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
