import functools
import logging

try:
    import torch
except ImportError:
    torch = None
import sparse
import scipy.sparse
import numpy as np
import cloudpickle

from libertem.io.dataset.base import DataTile, Partition
from .base import Job, Task, ResultTile
from libertem.masks import to_dense, to_sparse, is_sparse
from libertem.common import Slice
from libertem.common.buffers import zeros_aligned

log = logging.getLogger(__name__)


def _make_mask_slicer(computed_masks, dtype, sparse_backend, transpose):
    @functools.lru_cache(maxsize=None)
    def _get_masks_for_slice(slice_):
        stack_height = computed_masks.shape[0]
        m = slice_.get(computed_masks, sig_only=True)
        # We need the mask's signal dimension flattened
        m = m.reshape((stack_height, -1))
        if transpose:
            # We need the stack transposed in the next step
            m = m.T

        if is_sparse(m):
            if sparse_backend == 'sparse.pydata':
                # sparse.pydata.org is fastest for masks with few layers
                # and few entries
                return m.astype(dtype)
            elif 'scipy.sparse' in sparse_backend:
                # Just for calculation: scipy.sparse.csr_matrix is
                # the fastest for dot product of deep mask stack
                iis, jjs = m.coords
                values = m.data
                if sparse_backend == 'scipy.sparse.csc':
                    return scipy.sparse.csc_matrix((values, (iis, jjs)), shape=m.shape, dtype=dtype)
                else:
                    return scipy.sparse.csr_matrix((values, (iis, jjs)), shape=m.shape, dtype=dtype)
            else:
                raise ValueError(
                    "sparse_backend %s not implemented, can be 'scipy.sparse', "
                    "'scipy.sparse.csc' or 'sparse.pydata'" % sparse_backend)
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
        '''
        use_sparse can be None, True, 'scipy.sparse', 'scipy.sparse.csc' or 'sparse.pydata'
        '''
        super().__init__(*args, **kwargs)
        # Choose default back-end
        # If None, decide in the mask container
        if use_sparse is True:
            use_sparse = 'scipy.sparse'
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
    '''
    Container for mask stacks that are created from factory functions.

    It allows stacking, cached slicing, transposing and conversion
    to condition the masks for high-performance dot products.
    '''
    def __init__(self, mask_factories, dtype=None, use_sparse=None, count=None):
        '''
        use_sparse can be None, 'scipy.sparse', 'scipy.sparse.csc' or 'sparse.pydata'
        '''
        self.mask_factories = mask_factories
        # If we generate a whole mask stack with one function call,
        # we should know the length without generating the mask stack
        self._length = count
        self._dtype = dtype
        self._use_sparse = use_sparse
        self._mask_cache = {}
        # lazily initialized in the worker process, to keep task size small:
        self._computed_masks = None
        self._get_masks_for_slice = {}
        self.validate_mask_functions()

    def __getstate__(self):
        # don't even try to pickle mask cache
        state = self.__dict__
        state['_get_masks_for_slice'] = {}
        return state

    def validate_mask_functions(self):
        fns = self.mask_factories
        # 1 MB, magic number L3 cache
        limit = 2**20
        if callable(fns):
            fns = [fns]
        for fn in fns:
            s = len(cloudpickle.dumps(fn))
            if s > limit:
                log.warning(
                    'Mask factory size %s larger than warning limit %s, may be inefficient'
                    % (s, limit)
                )

    def __len__(self):
        if self._length is not None:
            return self._length
        elif not callable(self.mask_factories):
            return len(self.mask_factories)
        else:
            return len(self.computed_masks)

    def get(self, key, dtype=None, sparse_backend=None, transpose=True):
        if isinstance(key, Partition):
            slice_ = key.slice
        elif isinstance(key, DataTile):
            slice_ = key.tile_slice
        elif isinstance(key, Slice):
            slice_ = key
        else:
            raise TypeError(
                "MaskContainer.get() can only be called with "
                "DataTile/Slice/Partition instances"
            )
        return self.get_masks_for_slice(
            slice_.discard_nav(),
            dtype=dtype,
            sparse_backend=sparse_backend,
            transpose=transpose
        )

    @property
    def dtype(self):
        if self._dtype is None:
            return self.computed_masks.dtype
        else:
            return self._dtype

    @property
    def use_sparse(self):
        if self._use_sparse is None:
            # Computing the masks sets _use_sparse
            self.computed_masks
        return self._use_sparse

    def _compute_masks(self):
        """
        Call mask factories and combine to mask stack

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

        default_sparse = 'scipy.sparse'

        if callable(self.mask_factories):
            raw_masks = self.mask_factories()
            if not is_sparse(raw_masks):
                default_sparse = False
            mask_slices = [raw_masks]
        else:
            mask_slices = []
            for f in self.mask_factories:
                m = f()
                # Scipy.sparse is always 2D, so we have to convert here
                # before reshaping
                if scipy.sparse.issparse(m):
                    m = sparse.COO.from_scipy_sparse(m)
                # We reshape to be a stack of 1 so that we can unify code below
                m = m.reshape((1, ) + m.shape)
                if not is_sparse(m):
                    default_sparse = False
                mask_slices.append(m)

        if self._use_sparse is None:
            self._use_sparse = default_sparse

        if self.use_sparse:
            # Conversion to correct back-end will happen later
            # Use sparse.pydata because it implements the array interface
            # which makes mask handling easier
            masks = sparse.concatenate(
                [to_sparse(m) for m in mask_slices]
            )
        else:
            masks = np.concatenate(
                [to_dense(m) for m in mask_slices]
            )
        return masks

    def get_masks_for_slice(self, slice_, dtype=None, sparse_backend=None, transpose=True):
        if dtype is None:
            dtype = self.dtype
        if sparse_backend is None:
            sparse_backend = self.use_sparse
        key = (dtype, sparse_backend, transpose)
        if key not in self._get_masks_for_slice:
            self._get_masks_for_slice[key] = _make_mask_slicer(
                self.computed_masks,
                dtype=dtype,
                sparse_backend=sparse_backend,
                transpose=transpose
            )
        return self._get_masks_for_slice[key](slice_)

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
        use_torch :
            Setting to False disables torch. Setting to True doesn't enforce
            torch. Torch will be disabled if it is not installed
            or if the dtypes are unsuitable. It works only for float32 or float64,
            and only if both mask and data have the same dtype.
        dtype :
            dtype to use for the calculation and the result.
        """
        super().__init__(*args, **kwargs)
        self.masks = masks
        self.use_torch = use_torch
        self.dtype = np.dtype(dtype)
        self.read_dtype = self._input_dtype(self.partition.dtype)
        self.mask_dtype = self._input_dtype(self.masks.dtype)
        if torch is None or self.dtype.kind != 'f' or self.read_dtype != self.mask_dtype:
            self.use_torch = False

    def _input_dtype(self, dtype):
        '''
        Determine which dtype to request for masks or input data based on their native
        dtype and self.dtype.

        A dot product with floats is significantly faster than doing the same processing with
        integer data types, because Numpy uses its internal implementation of the dot product
        for integers, while it uses optimized libraries like OpenBLAS for floats. Furthermore,
        floats allow using torch.

        For that reason, we use floats by default. If floats are used, we request native integer
        mask data and native integer input data to be converted to floats already at the source.
        That helps to avoid an additional conversion step. As an example, the mask container can
        cache a float version with get_mask_for_slice(), and the K2IS reader can convert its 12 bit
        packed uints to floats directly.

        In case a conversion is requested, we decide if float32 or float64 is best suited based on
        the itemsize of the source data. In particular, float64 will be used for int64 data.

        In case the native dtype is not integer or in case the processing is not done with floating
        point numbers, return the native dtype.

        FIXME It should be tested in more detail which dtype combinations (32 bit vs 64 bit, complex
        vs real) are ideal for the dot product and what impact the conversion has on overall
        performance. In particular, the impact of size vs conversion overhead is not trivial to
        predict and might depend on the CPU type and load.
        The decision logic in this function should be adapted accordingly.
        '''
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
            if isinstance(masks, sparse.SparseArray):
                result = sparse.dot(flat_data, masks)
            elif scipy.sparse.issparse(masks):
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
