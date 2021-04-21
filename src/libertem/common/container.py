import functools
import logging

import sparse
import scipy.sparse
import numpy as np
import cloudpickle

from libertem.masks import to_dense, to_sparse, is_sparse
from libertem.common import Slice


log = logging.getLogger(__name__)


def _build_sparse(m, dtype, sparse_backend, backend):
    if sparse_backend == 'sparse.pydata' and backend == 'numpy':
        # sparse.pydata.org is fastest for masks with few layers
        # and few entries
        return m.astype(dtype)
    elif 'scipy.sparse' in sparse_backend:
        if backend == 'numpy':
            lib = scipy.sparse
        elif backend == 'cupy':
            # Avoid import if possible
            import cupyx.scipy.sparse
            lib = cupyx.scipy.sparse
        else:
            raise ValueError(
                f"Backend {backend} not supported for sparse_backend {sparse_backend}."
            )
        iis, jjs = m.coords
        values = m.data
        if sparse_backend == 'scipy.sparse.csc':
            s = scipy.sparse.csc_matrix(
                (values, (iis, jjs)), shape=m.shape, dtype=dtype)
            assert s.has_canonical_format
            return lib.csc_matrix(s)
        elif sparse_backend == 'scipy.sparse' or sparse_backend == 'scipy.sparse.csr':
            s = scipy.sparse.csr_matrix(
                (values, (iis, jjs)), shape=m.shape, dtype=dtype)
            assert s.has_canonical_format
            return lib.csr_matrix(s)
    # Fall through if no return statement was reached
    raise ValueError(
        f"sparse_backend {sparse_backend} not implemented for backend {backend}. "
        "Backend 'numpy' supports 'sparse.pydata', 'scipy.sparse', 'scipy.sparse.csc' or "
        "'scipy.sparse.csr'. "
        "Backend 'cupy' supports 'scipy.sparse', 'scipy.sparse.csc' or 'scipy.sparse.csr'. "
    )


def _make_mask_slicer(computed_masks, dtype, sparse_backend, transpose, backend):
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
            return _build_sparse(m, dtype, sparse_backend, backend)
        else:
            if backend == 'numpy':
                return m.astype(dtype)
            elif backend == 'cupy':
                # Avoid importing if possible
                import cupy
                return cupy.array(m.astype(dtype))
    return _get_masks_for_slice


class MaskContainer(object):
    '''
    Container for mask stacks that are created from factory functions.

    It allows stacking, cached slicing, transposing and conversion
    to condition the masks for high-performance dot products.

    use_sparse can be None, 'scipy.sparse', 'scipy.sparse.csc' or 'sparse.pydata'
    '''
    def __init__(self, mask_factories, dtype=None, use_sparse=None, count=None, backend=None):
        self.mask_factories = mask_factories
        # If we generate a whole mask stack with one function call,
        # we should know the length without generating the mask stack
        self._length = count
        self._dtype = dtype
        self._use_sparse = use_sparse
        self._mask_cache = {}
        # lazily initialized in the worker process, to keep task size small:
        self._computed_masks = None
        if backend is None:
            backend = 'numpy'
        self.backend = backend
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

    def get(self, key: Slice, dtype=None, sparse_backend=None, transpose=True, backend=None):
        if isinstance(key, Slice):
            slice_ = key
        else:
            raise TypeError(
                "MaskContainer.get() can only be called with "
                "DataTile/Slice/Partition instances"
            )
        if backend is None:
            backend = self.backend
        return self.get_masks_for_slice(
            slice_.discard_nav(),
            dtype=dtype,
            sparse_backend=sparse_backend,
            transpose=transpose,
            backend=backend
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

    def get_masks_for_slice(self, slice_, dtype=None, sparse_backend=None,
            transpose=True, backend='numpy'):
        if dtype is None:
            dtype = self.dtype
        if sparse_backend is None:
            sparse_backend = self.use_sparse
        if backend is None:
            backend = self.backend
        key = (dtype, sparse_backend, transpose, backend)
        if key not in self._get_masks_for_slice:
            self._get_masks_for_slice[key] = _make_mask_slicer(
                self.computed_masks,
                dtype=dtype,
                sparse_backend=sparse_backend,
                transpose=transpose,
                backend=backend
            )
        return self._get_masks_for_slice[key](slice_)

    @property
    def computed_masks(self):
        if self._computed_masks is None:
            self._computed_masks = self._compute_masks()
        return self._computed_masks
