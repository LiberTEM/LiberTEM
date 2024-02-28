import functools
import logging
from typing import Union, Callable, Optional
from collections.abc import Sequence
from typing_extensions import Literal

import sparse
import scipy.sparse
import numpy as np
import numpy.typing as npt
import cloudpickle
from sparseconverter import (
    CPU_BACKENDS, CUDA, CUPY_BACKENDS, for_backend, ArrayT, ArrayBackend, NUMPY
)
from libertem.io.dataset.base.tiling_scheme import TilingScheme

from libertem.common.sparse import to_dense, to_sparse, is_sparse
from libertem.common import Slice


log = logging.getLogger(__name__)

FactoryT = Callable[[], ArrayT]
SparseSupportedT = Literal[
    'sparse.pydata',
    'sparse.pydata.GCXS',
    'scipy.sparse',
    'scipy.sparse.csc',
    'scipy.sparse.csr',
]


def _build_sparse(m, dtype: npt.DTypeLike, sparse_backend: SparseSupportedT, backend: ArrayBackend):
    if sparse_backend == 'sparse.pydata' and backend == NUMPY:
        # sparse.pydata.org is fastest for masks with few layers
        # and few entries
        return m.astype(dtype)
    elif sparse_backend == 'sparse.pydata.GCXS' and backend == NUMPY:
        # sparse.pydata.org is fastest for masks with few layers
        # and few entries
        return sparse.GCXS(m.astype(dtype))
    elif 'scipy.sparse' in sparse_backend:
        if backend in CPU_BACKENDS or backend == CUDA:
            lib = scipy.sparse
        elif backend in CUPY_BACKENDS:
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
        "CPU-based backends supports 'sparse.pydata', 'sparse.pydata.GCXS', 'scipy.sparse', "
        "'scipy.sparse.csc' or 'scipy.sparse.csr'. "
        "CUDA-based backends supports 'scipy.sparse', 'scipy.sparse.csc' or 'scipy.sparse.csr'. "
    )


def _make_mask_slicer(
    computed_masks: ArrayT,
    dtype: npt.DTypeLike,
    sparse_backend: Union[Literal[False], SparseSupportedT],
    transpose: bool,
    backend: ArrayBackend,
):
    @functools.cache
    def _get_masks_for_slice(slice_):
        stack_height = computed_masks.shape[0]
        m = slice_.get(computed_masks, sig_only=True)
        # We need the mask's signal dimension flattened
        m = m.reshape((stack_height, -1))
        if transpose:
            # We need the stack transposed in the next step
            m = m.T
        if sparse_backend is False:
            return for_backend(m, backend).astype(dtype)
        else:
            return _build_sparse(m, dtype, sparse_backend, backend)
    return _get_masks_for_slice


class MaskContainer:
    '''
    Container for mask stacks that are created from factory functions.

    It allows stacking, cached slicing, transposing and conversion
    to condition the masks for high-performance dot products.

    Computation of masks is delayed until as late as possible,
    but is done automatically when necessary. Methods which can trigger
    mask instantiation include:

      - container.use_sparse
      - len(container) [if the count argument is None at __init__]
      - container.dtype [if the dtype argument is None at __init__]
      - any of the get() methods

    use_sparse at init can be None, False, True or any supported
    sparse backend as a string in {'scipy.sparse', 'scipy.sparse.csc',
    'scipy.sparse.csr', 'sparse.pydata', 'sparse.pydata.GCXS'}

    use_sparse as None means the sparse mode will be chosen only after
    the masks are instantiated. All masks being sparse will activate sparse
    processing using the backend in default_sparse, else dense processing
    will be used on the appropriate backend.
    '''
    def __init__(
        self,
        mask_factories: Union[FactoryT, Sequence[FactoryT]],
        dtype: Optional[npt.DTypeLike] = None,
        use_sparse: Optional[Union[bool, SparseSupportedT]] = None,
        count: Optional[int] = None,
        backend: Optional[ArrayBackend] = None,
        default_sparse: SparseSupportedT = 'scipy.sparse',
    ):
        self.mask_factories = mask_factories
        # If we generate a whole mask stack with one function call,
        # we should know the length without generating the mask stack
        self._length = count
        self._dtype = dtype
        self._mask_cache = {}
        # lazily initialized in the worker process, to keep task size small:
        self._computed_masks = None
        if backend is None:
            backend = 'numpy'
        self.backend = backend
        self._get_masks_for_slice = {}
        # from Python 3.8....
        # assert default_sparse in typing.get_args(SparseSupportedT)
        self._default_sparse = default_sparse
        self._use_sparse: Union[Literal[False], None, SparseSupportedT]
        # Try to resolve if we are actually using sparse upfront,
        # this is not always possible as it depends on whether the
        # mask_factories will all return sparse matrices
        if use_sparse is True:
            self._use_sparse = default_sparse
        elif use_sparse is False:
            self._use_sparse = False
        elif isinstance(use_sparse, str) and (
            # This should be rendered compatible with SPARSE_BACKENDS frozenset
            # but there are issues of capitalization and naming
            use_sparse.lower().startswith('scipy.sparse')
            or use_sparse.lower().startswith('sparse.pydata')
        ):
            self._use_sparse = use_sparse
        elif use_sparse is None:
            # User doesn't specify, will use sparse if masks
            # are sparse and we are on a compatible backend
            if (
                default_sparse.startswith('sparse.pydata')
                and self.backend in CUPY_BACKENDS
            ):
                # sparse.pydata cannot run on CuPy, so densify to allow calculation
                self._use_sparse = False
            else:
                # we can't determine _use_sparse without creating the masks
                # themselves and we want to delay this as late as possible
                # leave as None for now and resolve on first access to
                # the self.use_sparse property
                self._use_sparse = None
        else:
            raise ValueError(f'use_sparse not an allowed value: {use_sparse}')

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

    def get_for_idx(self, scheme: TilingScheme, idx: int, *args, **kwargs):
        slice_ = scheme[idx]
        return self._get(slice_, *args, **kwargs)

    def get_for_sig_slice(self, sig_slice: Slice, *args, **kwargs):
        """
        Same as `get`, but without calling `discard_nav()` on the slice
        """
        return self._get(sig_slice, *args, **kwargs)

    def get(self, key: Slice, dtype=None, sparse_backend=None, transpose=True, backend=None):
        if not isinstance(key, Slice):
            raise TypeError(
                "MaskContainer.get() can only be called with "
                "DataTile/Slice/Partition instances"
            )
        return self._get(key.discard_nav(), dtype, sparse_backend, transpose, backend)

    def _get(self, slice_: Slice, dtype=None, sparse_backend=None, transpose=True, backend=None):
        if backend is None:
            backend = self.backend
        return self.get_masks_for_slice(
            slice_,
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
    def use_sparse(self) -> Union[SparseSupportedT, Literal[False]]:
        # As far as possible use_sparse was resolved at __init__
        # but if we don't know if the masks are sparse we may still arrive
        # here with self._use_sparse is None
        if self._use_sparse is None:
            if is_sparse(self.computed_masks):
                # The first time the condition is hit will cause
                # mask computation but on subsequent tries we will
                # fall through to the normal return
                self._use_sparse = self._default_sparse
            else:
                self._use_sparse = False
        return self._use_sparse

    def _compute_masks(self) -> Union[np.ndarray, sparse.COO, sparse.GCXS]:
        """
        Call mask factories and combine into a mask stack

        Uses the internal attr self._use_sparse, which could be None
        if we were unable to resolve the sparse mode at __init__
        If self._use_sparse is None and all masks are sparse then will
        return a sparse stack else return a dense stack
        Otherwise if self._use_sparse is simply False then return
        dense, anything else return as a sparse stack

        Returns
        -------
        an array-like mask stack with contents as they were
        created by the factories
        """
        mask_slices = []
        if callable(self.mask_factories):
            raw_masks = self.mask_factories()
            mask_slices.append(raw_masks)
        else:
            for f in self.mask_factories:
                m = f()
                # Scipy.sparse is always 2D, so we have to convert here
                # before reshaping
                if scipy.sparse.issparse(m):
                    m = sparse.COO.from_scipy_sparse(m)
                # We reshape to be a stack of 1 so that we can unify code below
                m = m.reshape((1, ) + m.shape)
                mask_slices.append(m)

        # Fully resolve _use_sparse based on sparsity of masks.
        # The return type (sparse or dense) from this function
        # is used to resolve _use_sparse permanently in the
        # self.use_sparse property method
        masks_are_sparse = all(is_sparse(m) for m in mask_slices)
        use_sparse = self._use_sparse
        if use_sparse is None:
            if masks_are_sparse:
                use_sparse = self._default_sparse
            else:
                use_sparse = False

        if use_sparse is not False:
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
