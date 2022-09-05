from functools import partial, reduce
import itertools
import time
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp
import sparse


NUMPY = 'numpy'
# Return type of aggregation operations on scipy.sparse matrices
# Including this here allows UDF.forbuf to work.
NUMPY_MATRIX = 'numpy.matrix'
CUDA = 'cuda'  # NumPy input, but on GPU device -- UDF does own transfer
CUPY = 'cupy'
SPARSE_COO = 'sparse.COO'
SPARSE_GCXS = 'sparse.GCXS'
SPARSE_DOK = 'sparse.DOK'

# On Python 3.6 only the matrix interface of SciPy is supported
SCIPY_COO = 'scipy.sparse.coo_matrix'
SCIPY_CSR = 'scipy.sparse.csr_matrix'
SCIPY_CSC = 'scipy.sparse.csc_matrix'

CUPY_SCIPY_COO = 'cupyx.scipy.sparse.coo_matrix'
CUPY_SCIPY_CSR = 'cupyx.scipy.sparse.csr_matrix'
CUPY_SCIPY_CSC = 'cupyx.scipy.sparse.csc_matrix'

CPU_BACKENDS = frozenset((
    NUMPY, NUMPY_MATRIX, SPARSE_COO, SPARSE_GCXS, SPARSE_DOK, SCIPY_COO, SCIPY_CSR, SCIPY_CSC
))
CUPY_BACKENDS = frozenset((CUPY, CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC))
# "on CUDA, but no CuPy" backend that receives NumPy arrays
CUDA_BACKENDS = CUPY_BACKENDS.union((CUDA, ))
BACKENDS = CPU_BACKENDS.union(CUDA_BACKENDS)
# Backends that support n-dimensional arrays as opposed to 2D-only
ND_BACKENDS = frozenset((NUMPY, CUDA, CUPY, SPARSE_COO, SPARSE_GCXS, SPARSE_DOK))
# 2D backends
D2_BACKENDS = frozenset((
    NUMPY_MATRIX, SCIPY_COO, SCIPY_CSR, SCIPY_CSC, CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC
))
# Array classes that are not ND and not 2D are thinkable, i.e. automatically assuming
# a new class falls in these two categories may be wrong.
# Keep this consistency check in case arrays are added
assert ND_BACKENDS.union(D2_BACKENDS) == BACKENDS

SPARSE_BACKENDS = frozenset((
    SPARSE_COO, SPARSE_GCXS, SPARSE_DOK,
    SCIPY_COO, SCIPY_CSR, SCIPY_CSC,
    CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC
))
DENSE_BACKENDS = BACKENDS - SPARSE_BACKENDS

ArrayBackend = Any


class ClassDict:
    _classes = {
        NUMPY_MATRIX: np.matrix,
        NUMPY: np.ndarray,
        CUDA: np.ndarray,
        SPARSE_COO: sparse.COO,
        SPARSE_GCXS: sparse.GCXS,
        SPARSE_DOK: sparse.DOK,
        SCIPY_COO: sp.coo_matrix,
        SCIPY_CSR: sp.csr_matrix,
        SCIPY_CSC: sp.csc_matrix
    }

    def __getitem__(self, item):
        res = self._classes.get(item, None)
        if res is None:
            return self._get_lazy(item)
        else:
            return res

    def _get_lazy(self, item):
        if item == CUPY:
            import cupy
            res = cupy.ndarray
        elif item in (CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC):
            import cupyx.scipy  # noqa: F401
            res = eval(item)
        else:
            raise KeyError(f'Unknown format {item}')
        self._classes[item] = res
        return res


classes = ClassDict()

base_dtypes = {
    NUMPY: bool,
    NUMPY_MATRIX: bool,
    CUDA: bool,
    CUPY: bool,
    SPARSE_COO: bool,
    SPARSE_GCXS: bool,
    SPARSE_DOK: bool,
    SCIPY_COO: bool,
    SCIPY_CSR: bool,
    SCIPY_CSC: bool,
    CUPY_SCIPY_COO: np.float32,
    CUPY_SCIPY_CSC: np.float32,
    CUPY_SCIPY_CSR: np.float32,
}


def prod(shape):
    # Force 64 bit since int on Windows is 32 bit, leading to possible
    # overflows of np.prod(shape)
    return np.prod(shape, dtype=np.int64)


def _flatsig(arr):
    '''
    Convert to 2D for formats that only support two dimensions.

    All dimensions except the first one are flattened.
    '''
    target_shape = (arr.shape[0], prod(arr.shape[1:]))
    # print('_flatsig', arr.shape, type(arr), target_shape)
    return arr.reshape(target_shape)


def _identity(arr):
    # print('_identity', arr.shape, type(arr))
    return arr


def _GCXS_to_coo(arr: sparse.GCXS):
    reshaped = arr.reshape((arr.shape[0], -1))
    return reshaped.to_scipy_sparse().asformat('coo')


def _GCXS_to_csr(arr: sparse.GCXS):
    reshaped = arr.reshape((arr.shape[0], -1))
    return reshaped.to_scipy_sparse().asformat('csr')


def _GCXS_to_csc(arr: sparse.GCXS):
    reshaped = arr.reshape((arr.shape[0], -1))
    return reshaped.to_scipy_sparse().asformat('csc')


def chain(*functions):
    '''
    Create a function G(x) = f3(f2(f1(x)))
    from functions (f1, f2, f3)
    '''
    assert len(functions) >= 1
    return reduce(
        lambda val, func: (lambda x: func(val(x))),
        functions[1:],
        functions[0]
    )


class ConverterDict:
    def __init__(self):
        self._converters = {}
        for backend in BACKENDS:
            self._converters[(backend, backend)] = _identity
        # Both are NumPy arrays, distinguished for device selection
        for left in (NUMPY, CUDA):
            for right in (NUMPY, CUDA):
                self._converters[(left, right)] = _identity
        self._converters[(NUMPY, NUMPY_MATRIX)] = chain(_flatsig, np.matrix)
        self._converters[(NUMPY_MATRIX, NUMPY)] = np.array
        # Support direct construction from each other
        for left in (
                    NUMPY, CUDA, SPARSE_COO, SPARSE_GCXS, SPARSE_DOK,
                    SCIPY_COO, SCIPY_CSR, SCIPY_CSC
                ):
            for right in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                if (left, right) not in self._converters:
                    self._converters[(left, right)] = classes[right]
        # Overwrite from before
        self._converters[(SPARSE_DOK, SPARSE_GCXS)] = partial(sparse.DOK.asformat, format='gcxs')
        self._converters[(SPARSE_GCXS, SPARSE_DOK)] = partial(sparse.GCXS.asformat, format='dok')

        for left in NUMPY, CUDA, SCIPY_COO, SCIPY_CSR, SCIPY_CSC:
            for right in SCIPY_COO, SCIPY_CSR, SCIPY_CSC:
                if (left, right) not in self._converters:
                    self._converters[(left, right)] = chain(_flatsig, classes[right])
        for left in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
            for right in NUMPY, CUDA:
                if (left, right) not in self._converters:
                    self._converters[(left, right)] = classes[left].todense
        for left in SCIPY_COO, SCIPY_CSR, SCIPY_CSC:
            for right in NUMPY, CUDA:
                if (left, right) not in self._converters:
                    self._converters[(left, right)] = classes[left].toarray
        for left in SCIPY_COO, SCIPY_CSR, SCIPY_CSC:
            for right in SPARSE_COO, SPARSE_GCXS, SPARSE_DOK:
                if (left, right) not in self._converters:
                    self._converters[(left, right)] = classes[right].from_scipy_sparse
        self._converters[(SPARSE_COO, SCIPY_COO)] = chain(_flatsig, sparse.COO.to_scipy_sparse)
        self._converters[(SPARSE_COO, SCIPY_CSR)] = chain(_flatsig, sparse.COO.tocsr)
        self._converters[(SPARSE_COO, SCIPY_CSC)] = chain(_flatsig, sparse.COO.tocsc)
        self._converters[(SPARSE_GCXS, SCIPY_COO)] = _GCXS_to_coo
        self._converters[(SPARSE_GCXS, SCIPY_CSR)] = _GCXS_to_csr
        self._converters[(SPARSE_GCXS, SCIPY_CSC)] = _GCXS_to_csc

        for right in SCIPY_COO, SCIPY_CSR, SCIPY_CSC:
            if (SPARSE_DOK, right) not in self._converters:
                self._converters[(SPARSE_DOK, right)] = chain(
                    self._converters[(SPARSE_DOK, SPARSE_COO)],
                    self._converters[(SPARSE_COO, right)]
                )

        for left in CPU_BACKENDS:
            proxy = NUMPY
            right = NUMPY_MATRIX
            if (left, right) not in self._converters:
                c1 = self._converters[(left, proxy)]
                c2 = self._converters[(proxy, right)]
                self._converters[(left, right)] = chain(c1, c2)
        for right in CPU_BACKENDS:
            proxy = NUMPY
            left = NUMPY_MATRIX
            if (left, right) not in self._converters:
                c1 = self._converters[(left, proxy)]
                c2 = self._converters[(proxy, right)]
                self._converters[(left, right)] = chain(c1, c2)

    def _populate_cupy(self):
        import cupy
        import cupyx.scipy

        def detect_csr_complex128_bug():
            # https://github.com/cupy/cupy/issues/7035
            a = cupy.array([
                (1, 1j),
                (1j, -1-1j)

            ]).astype(np.complex128)
            sp = cupyx.scipy.sparse.csr_matrix(a)
            return not np.allclose(a, sp.todense())

        has_csr_complex128_bug = detect_csr_complex128_bug()

        CUPY_SPARSE_DTYPES = {
            np.float32, np.float64, np.complex64, np.complex128
        }

        def _GCXS_to_cupy_coo(arr: sparse.GCXS):
            reshaped = arr.reshape((arr.shape[0], -1))
            return cupyx.scipy.sparse.coo_matrix(reshaped.to_scipy_sparse())

        def _GCXS_to_cupy_csr(arr: sparse.GCXS):
            reshaped = arr.reshape((arr.shape[0], -1))
            return cupyx.scipy.sparse.csr_matrix(reshaped.to_scipy_sparse())

        def _GCXS_to_cupy_csc(arr: sparse.GCXS):
            reshaped = arr.reshape((arr.shape[0], -1))
            return cupyx.scipy.sparse.csc_matrix(reshaped.to_scipy_sparse())

        def _GCXS_to_cupy(arr: sparse.GCXS):
            reshaped = arr.reshape((arr.shape[0], -1))
            # Avoid changing the compressed axes
            if arr.compressed_axes == (0, ):
                return cupyx.scipy.sparse.csr_matrix(reshaped.to_scipy_sparse()).get()
            elif arr.compressed_axes == (1, ):
                return cupyx.scipy.sparse.csc_matrix(reshaped.to_scipy_sparse()).get()
            else:
                raise RuntimeError('Unexpected compressed axes in GCXS')

        def _CUPY_to_scipy_coo(arr: cupy.ndarray):
            reshaped = arr.reshape((arr.shape[0], -1))
            if arr.dtype in CUPY_SPARSE_DTYPES:
                intermediate = cupyx.scipy.sparse.coo_matrix(reshaped)
                return intermediate.get()
            else:
                intermediate = cupy.asnumpy(reshaped)
                return sp.coo_matrix(intermediate)

        def _CUPY_to_scipy_csr(arr: cupy.ndarray):
            reshaped = arr.reshape((arr.shape[0], -1))
            if arr.dtype in CUPY_SPARSE_DTYPES:
                intermediate = cupyx.scipy.sparse.csr_matrix(reshaped)
                return intermediate.get()
            else:
                intermediate = cupy.asnumpy(reshaped)
                return sp.csr_matrix(intermediate)

        def _CUPY_to_scipy_csc(arr: cupy.ndarray):
            reshaped = arr.reshape((arr.shape[0], -1))
            if arr.dtype in CUPY_SPARSE_DTYPES:
                intermediate = cupyx.scipy.sparse.csc_matrix(reshaped)
                return intermediate.get()
            else:
                intermediate = cupy.asnumpy(reshaped)
                return sp.csc_matrix(intermediate)

        def _CUPY_to_sparse_coo(arr: cupy.ndarray):
            if arr.dtype in CUPY_SPARSE_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                intermediate = cupyx.scipy.sparse.coo_matrix(reshaped)
                return sparse.COO(intermediate.get()).reshape(arr.shape)
            else:
                intermediate = cupy.asnumpy(arr)
                return sparse.COO.from_numpy(intermediate)

        def _CUPY_to_sparse_gcxs(arr: cupy.ndarray):
            if arr.dtype in CUPY_SPARSE_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                intermediate = cupyx.scipy.sparse.csr_matrix(reshaped)
                return sparse.GCXS(intermediate.get()).reshape(arr.shape)
            else:
                intermediate = cupy.asnumpy(arr)
                return sparse.GCXS.from_numpy(intermediate)

        def _CUPY_to_sparse_dok(arr: cupy.ndarray):
            if arr.dtype in CUPY_SPARSE_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                intermediate = cupyx.scipy.sparse.coo_matrix(reshaped)
                return sparse.DOK(intermediate.get()).reshape(arr.shape)
            else:
                intermediate = cupy.asnumpy(arr)
                return sparse.DOK.from_numpy(intermediate)

        def _sparse_coo_to_CUPY(arr: sparse.COO):
            if arr.dtype in CUPY_SPARSE_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                intermediate = cupyx.scipy.sparse.coo_matrix(reshaped.to_scipy_sparse())
                return intermediate.toarray().reshape(arr.shape)
            else:
                intermediate = arr.todense()
                return cupy.array(intermediate)

        def _sparse_gcxs_to_CUPY(arr: sparse.GCXS):
            if arr.dtype in CUPY_SPARSE_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                if arr.compressed_axes == (0, ):
                    intermediate = cupyx.scipy.sparse.csr_matrix(reshaped.to_scipy_sparse())
                elif arr.compressed_axes == (1, ):
                    intermediate = cupyx.scipy.sparse.csc_matrix(reshaped.to_scipy_sparse())
                return intermediate.toarray().reshape(arr.shape)
            else:
                intermediate = arr.todense()
                return cupy.array(intermediate)

        def _sparse_dok_to_CUPY(arr: sparse.DOK):
            if arr.dtype in CUPY_SPARSE_DTYPES:
                reshaped = arr.reshape((arr.shape[0], -1))
                intermediate = cupyx.scipy.sparse.coo_matrix(reshaped.to_coo().to_scipy_sparse())
                return intermediate.toarray().reshape(arr.shape)
            else:
                intermediate = arr.todense()
                return cupy.array(intermediate)

        def _adjust_dtype_cupy_sparse(arr):
            if arr.dtype in CUPY_SPARSE_DTYPES:
                res = arr
                # print('_adjust_dtype_cupy_sparse passthrough', arr.dtype, res.dtype)
            else:
                # Base dtype is the same for all cupyx.scipy.sparse matrices
                res = arr.astype(np.result_type(arr, base_dtypes[CUPY_SCIPY_COO]))
                # print('_adjust_dtype_cupy_sparse convert', arr.dtype, res.dtype)
            return res

        self._converters[(NUMPY, CUPY)] = cupy.array
        self._converters[(CUDA, CUPY)] = cupy.array
        self._converters[(CUPY, NUMPY)] = cupy.asnumpy
        self._converters[(CUPY, CUDA)] = cupy.asnumpy
        # Accepted by constructor of target class
        for left in (SCIPY_COO, SCIPY_CSR, SCIPY_CSC,
                CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC):
            for right in CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC:
                if (left, right) not in self._converters:
                    if left in ND_BACKENDS:
                        self._converters[(left, right)] = chain(
                            _flatsig, _adjust_dtype_cupy_sparse, classes[right]
                        )
                    else:
                        self._converters[(left, right)] = chain(
                            _adjust_dtype_cupy_sparse, classes[right]
                        )
        # Work around https://github.com/cupy/cupy/issues/7035
        # Otherwise CUPY -> {CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC}
        # could be handled by the block above.
        left, right = CUPY, CUPY_SCIPY_COO
        if (left, right) not in self._converters:
            self._converters[(left, right)] = chain(
                _flatsig, _adjust_dtype_cupy_sparse, classes[right]
            )
        for right in CUPY_SCIPY_CSR, CUPY_SCIPY_CSC:
            if (left, right) not in self._converters:
                if has_csr_complex128_bug:
                    self._converters[(left, right)] = chain(
                        # First convert to COO which is not affected
                        # Fortunately the overhead is not too bad.
                        _flatsig, _adjust_dtype_cupy_sparse, classes[CUPY_SCIPY_COO], classes[right]
                    )
                else:
                    self._converters[(left, right)] = chain(

                        _flatsig, _adjust_dtype_cupy_sparse, classes[right]
                    )
        for left in NUMPY, CUDA:
            for right in CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC:
                if (left, right) not in self._converters:
                    c1 = self._converters[(left, CUPY)]
                    c2 = self._converters[(CUPY, right)]
                    self._converters[(left, right)] = chain(c1, c2)
        for left in CUPY_SCIPY_COO, CUPY_SCIPY_CSR, CUPY_SCIPY_CSC:
            if (left, CUPY) not in self._converters:
                self._converters[(left, CUPY)] = classes[left].toarray

        for (left, right) in [
                    (CUPY_SCIPY_COO, SCIPY_COO),
                    (CUPY_SCIPY_CSR, SCIPY_CSR),
                    (CUPY_SCIPY_CSC, SCIPY_CSC)
                ]:
            if (left, right) not in self._converters:
                self._converters[(left, right)] = classes[left].get

        self._converters[(SPARSE_GCXS, CUPY_SCIPY_COO)] = chain(
            _adjust_dtype_cupy_sparse, _GCXS_to_cupy_coo
        )
        self._converters[(SPARSE_GCXS, CUPY_SCIPY_CSR)] = chain(
            _adjust_dtype_cupy_sparse, _GCXS_to_cupy_csr
        )
        self._converters[(SPARSE_GCXS, CUPY_SCIPY_CSC)] = chain(
            _adjust_dtype_cupy_sparse, _GCXS_to_cupy_csc
        )
        self._converters[(SPARSE_GCXS, CUPY)] = _GCXS_to_cupy

        self._converters[(CUPY, SCIPY_COO)] = _CUPY_to_scipy_coo
        self._converters[(CUPY, SCIPY_CSR)] = _CUPY_to_scipy_csr
        self._converters[(CUPY, SCIPY_CSC)] = _CUPY_to_scipy_csc

        self._converters[(CUPY, SPARSE_COO)] = _CUPY_to_sparse_coo
        self._converters[(CUPY, SPARSE_GCXS)] = _CUPY_to_sparse_gcxs
        self._converters[(CUPY, SPARSE_DOK)] = _CUPY_to_sparse_dok

        self._converters[(SPARSE_COO, CUPY)] = _sparse_coo_to_CUPY
        self._converters[(SPARSE_GCXS, CUPY)] = _sparse_gcxs_to_CUPY
        self._converters[(SPARSE_DOK, CUPY)] = _sparse_dok_to_CUPY

        proxies = [
            (CUPY_SCIPY_COO, SCIPY_COO, NUMPY),
            (CUPY_SCIPY_COO, SCIPY_COO, CUDA),
            (CUPY_SCIPY_COO, CUPY_SCIPY_CSR, SCIPY_CSR),
            (CUPY_SCIPY_COO, CUPY_SCIPY_CSC, SCIPY_CSC),
            (CUPY_SCIPY_COO, SCIPY_COO, SPARSE_COO),
            (CUPY_SCIPY_COO, SCIPY_CSR, SPARSE_GCXS),
            (CUPY_SCIPY_COO, SCIPY_COO, SPARSE_DOK),

            (CUPY_SCIPY_CSR, SCIPY_CSR, NUMPY),
            (CUPY_SCIPY_CSR, SCIPY_CSR, CUDA),
            (CUPY_SCIPY_CSR, CUPY_SCIPY_COO, SCIPY_COO),
            (CUPY_SCIPY_CSR, CUPY_SCIPY_CSC, SCIPY_CSC),
            (CUPY_SCIPY_CSR, CUPY_SCIPY_COO, SPARSE_COO),
            (CUPY_SCIPY_CSR, SCIPY_CSR, SPARSE_GCXS),
            (CUPY_SCIPY_CSR, SCIPY_CSR, SPARSE_DOK),

            (CUPY_SCIPY_CSC, SCIPY_CSC, NUMPY),
            (CUPY_SCIPY_CSC, SCIPY_CSC, CUDA),
            (CUPY_SCIPY_CSC, CUPY_SCIPY_COO, SCIPY_COO),
            (CUPY_SCIPY_CSC, CUPY_SCIPY_CSR, SCIPY_CSR),
            (CUPY_SCIPY_CSC, CUPY_SCIPY_COO, SPARSE_COO),
            (CUPY_SCIPY_CSC, SCIPY_CSC, SPARSE_GCXS),
            (CUPY_SCIPY_CSC, SCIPY_CSC, SPARSE_DOK),

            (SCIPY_COO, CUPY_SCIPY_COO, CUPY),
            (SCIPY_CSR, CUPY_SCIPY_CSR, CUPY),
            (SCIPY_CSC, CUPY_SCIPY_CSC, CUPY),

            (SPARSE_COO, SCIPY_COO, CUPY_SCIPY_COO),
            (SPARSE_COO, SCIPY_COO, CUPY_SCIPY_CSR),
            (SPARSE_COO, SCIPY_COO, CUPY_SCIPY_CSC),
            (SPARSE_COO, SCIPY_COO, CUPY),

            (SPARSE_DOK, SCIPY_COO, CUPY_SCIPY_COO),
            (SPARSE_DOK, SCIPY_COO, CUPY_SCIPY_CSR),
            (SPARSE_DOK, SCIPY_COO, CUPY_SCIPY_CSC),
            (SPARSE_DOK, SCIPY_COO, CUPY),
        ]
        for left, proxy, right in proxies:
            if (left, right) not in self._converters:
                c1 = self._converters[(left, proxy)]
                c2 = self._converters[(proxy, right)]
                self._converters[(left, right)] = chain(c1, c2)

        for left in BACKENDS:
            proxy = NUMPY
            right = NUMPY_MATRIX
            if (left, right) not in self._converters:
                c1 = self._converters[(left, proxy)]
                c2 = self._converters[(proxy, right)]
                self._converters[(left, right)] = chain(c1, c2)
        for right in BACKENDS:
            proxy = NUMPY
            left = NUMPY_MATRIX
            if (left, right) not in self._converters:
                c1 = self._converters[(left, proxy)]
                c2 = self._converters[(proxy, right)]
                self._converters[(left, right)] = chain(c1, c2)

        for left in BACKENDS:
            for right in BACKENDS:
                if (left, right) not in self._converters:
                    raise RuntimeError(f'Missing converter {left} -> {right}')

    def __getitem__(self, item):
        res = self._converters.get(item, False)
        if res is False:
            left, right = item
            if left in CUDA_BACKENDS or right in CUDA_BACKENDS:
                self._populate_cupy()
            return self._converters[item]
        else:
            return res

    def get(self, item, default):
        try:
            return self.__getitem__(item)
        except KeyError:
            return default


converters = ConverterDict()
# Measured with a run of benchmark_conversions on a "representative machine" aka my laptop
# (32, 512, 512), np.float32, density 0.1
# See prototypes/array_formats/benchmark.ipynb to generate this list
# FIXME improve cost basis based on real use cases.
cost = {
    (SCIPY_CSC, SCIPY_CSC): 6.8e-07,
    (SCIPY_CSC, SPARSE_DOK): 0.79008641,
    (SCIPY_CSC, CUPY_SCIPY_COO): 0.01557962,
    (SCIPY_CSC, SCIPY_CSR): 0.00550373,
    (SCIPY_CSC, CUPY): 0.07368527,
    (SCIPY_CSC, CUDA): 0.00645836,
    (SCIPY_CSC, NUMPY): 0.00646419,
    (SCIPY_CSC, CUPY_SCIPY_CSR): 0.00894698,
    (SCIPY_CSC, NUMPY_MATRIX): 0.06175063,
    (SCIPY_CSC, SCIPY_COO): 0.00913096,
    (SCIPY_CSC, SPARSE_GCXS): 1.715e-05,
    (SCIPY_CSC, SPARSE_COO): 0.10771111,
    (SCIPY_CSC, CUPY_SCIPY_CSC): 0.00276772,
    (SPARSE_DOK, SCIPY_CSC): 0.42924186,
    (SPARSE_DOK, SPARSE_DOK): 8.1e-07,
    (SPARSE_DOK, CUPY_SCIPY_COO): 0.45422632,
    (SPARSE_DOK, SCIPY_CSR): 0.41378311,
    (SPARSE_DOK, CUPY): 0.12193486,
    (SPARSE_DOK, CUDA): 0.11697014,
    (SPARSE_DOK, NUMPY): 0.111815,
    (SPARSE_DOK, CUPY_SCIPY_CSR): 0.45438094,
    (SPARSE_DOK, NUMPY_MATRIX): 0.12303009,
    (SPARSE_DOK, SCIPY_COO): 0.40668993,
    (SPARSE_DOK, SPARSE_GCXS): 0.42601204,
    (SPARSE_DOK, SPARSE_COO): 0.39792751,
    (SPARSE_DOK, CUPY_SCIPY_CSC): 0.42898774,
    (CUPY_SCIPY_COO, SCIPY_CSC): 0.0520641,
    (CUPY_SCIPY_COO, SPARSE_DOK): 0.69566436,
    (CUPY_SCIPY_COO, CUPY_SCIPY_COO): 8e-07,
    (CUPY_SCIPY_COO, SCIPY_CSR): 0.05163085,
    (CUPY_SCIPY_COO, CUPY): 0.11410962,
    (CUPY_SCIPY_COO, CUDA): 0.01298642,
    (CUPY_SCIPY_COO, NUMPY): 0.01314296,
    (CUPY_SCIPY_COO, CUPY_SCIPY_CSR): 0.04786648,
    (CUPY_SCIPY_COO, NUMPY_MATRIX): 0.01981244,
    (CUPY_SCIPY_COO, SCIPY_COO): 0.00711411,
    (CUPY_SCIPY_COO, SPARSE_GCXS): 0.0518312,
    (CUPY_SCIPY_COO, SPARSE_COO): 0.0226392,
    (CUPY_SCIPY_COO, CUPY_SCIPY_CSC): 0.04807331,
    (SCIPY_CSR, SCIPY_CSC): 0.01132071,
    (SCIPY_CSR, SPARSE_DOK): 0.69419015,
    (SCIPY_CSR, CUPY_SCIPY_COO): 0.03507685,
    (SCIPY_CSR, SCIPY_CSR): 8.1e-07,
    (SCIPY_CSR, CUPY): 0.0736361,
    (SCIPY_CSR, CUDA): 0.00702799,
    (SCIPY_CSR, NUMPY): 0.00613211,
    (SCIPY_CSR, CUPY_SCIPY_CSR): 0.00214472,
    (SCIPY_CSR, NUMPY_MATRIX): 0.01560373,
    (SCIPY_CSR, SCIPY_COO): 0.00780115,
    (SCIPY_CSR, SPARSE_GCXS): 1.791e-05,
    (SCIPY_CSR, SPARSE_COO): 0.01731952,
    (SCIPY_CSR, CUPY_SCIPY_CSC): 0.01247452,
    (CUPY, SCIPY_CSC): 0.08765124,
    (CUPY, SPARSE_DOK): 0.44793429,
    (CUPY, CUPY_SCIPY_COO): 0.01578986,
    (CUPY, SCIPY_CSR): 0.08288962,
    (CUPY, CUPY): 8.1e-07,
    (CUPY, CUDA): 0.011692,
    (CUPY, NUMPY): 0.01315366,
    (CUPY, CUPY_SCIPY_CSR): 0.01067939,
    (CUPY, NUMPY_MATRIX): 0.01975819,
    (CUPY, SCIPY_COO): 0.07855287,
    (CUPY, SPARSE_GCXS): 0.09928033,
    (CUPY, SPARSE_COO): 0.07420958,
    (CUPY, CUPY_SCIPY_CSC): 0.01580088,
    (CUDA, SCIPY_CSC): 0.07511825,
    (CUDA, SPARSE_DOK): 0.42874134,
    (CUDA, CUPY_SCIPY_COO): 0.02223272,
    (CUDA, SCIPY_CSR): 0.07167406,
    (CUDA, CUPY): 0.0062838,
    (CUDA, CUDA): 7.5e-07,
    (CUDA, NUMPY): 8e-07,
    (CUDA, CUPY_SCIPY_CSR): 0.02201833,
    (CUDA, NUMPY_MATRIX): 0.00721113,
    (CUDA, SCIPY_COO): 0.06756767,
    (CUDA, SPARSE_GCXS): 0.08923092,
    (CUDA, SPARSE_COO): 0.05519825,
    (CUDA, CUPY_SCIPY_CSC): 0.03305383,
    (NUMPY, SCIPY_CSC): 0.07595573,
    (NUMPY, SPARSE_DOK): 0.42557287,
    (NUMPY, CUPY_SCIPY_COO): 0.03367578,
    (NUMPY, SCIPY_CSR): 0.07286735,
    (NUMPY, CUPY): 0.00426659,
    (NUMPY, CUDA): 8.1e-07,
    (NUMPY, NUMPY): 7.6e-07,
    (NUMPY, CUPY_SCIPY_CSR): 0.02202718,
    (NUMPY, NUMPY_MATRIX): 0.00715483,
    (NUMPY, SCIPY_COO): 0.06798563,
    (NUMPY, SPARSE_GCXS): 0.08982199,
    (NUMPY, SPARSE_COO): 0.05319294,
    (NUMPY, CUPY_SCIPY_CSC): 0.03394102,
    (CUPY_SCIPY_CSR, SCIPY_CSC): 0.00628662,
    (CUPY_SCIPY_CSR, SPARSE_DOK): 0.6935086,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_COO): 0.01812139,
    (CUPY_SCIPY_CSR, SCIPY_CSR): 0.00290519,
    (CUPY_SCIPY_CSR, CUPY): 0.07112426,
    (CUPY_SCIPY_CSR, CUDA): 0.0088542,
    (CUPY_SCIPY_CSR, NUMPY): 0.00929009,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_CSR): 8.4e-07,
    (CUPY_SCIPY_CSR, NUMPY_MATRIX): 0.01770824,
    (CUPY_SCIPY_CSR, SCIPY_COO): 0.01478172,
    (CUPY_SCIPY_CSR, SPARSE_GCXS): 0.00313166,
    (CUPY_SCIPY_CSR, SPARSE_COO): 0.03076698,
    (CUPY_SCIPY_CSR, CUPY_SCIPY_CSC): 0.00219811,
    (NUMPY_MATRIX, SCIPY_CSC): 0.08562059,
    (NUMPY_MATRIX, SPARSE_DOK): 0.38622755,
    (NUMPY_MATRIX, CUPY_SCIPY_COO): 0.04382591,
    (NUMPY_MATRIX, SCIPY_CSR): 0.08020001,
    (NUMPY_MATRIX, CUPY): 0.00930196,
    (NUMPY_MATRIX, CUDA): 0.00666705,
    (NUMPY_MATRIX, NUMPY): 0.00819635,
    (NUMPY_MATRIX, CUPY_SCIPY_CSR): 0.0261312,
    (NUMPY_MATRIX, NUMPY_MATRIX): 6.5e-07,
    (NUMPY_MATRIX, SCIPY_COO): 0.07484995,
    (NUMPY_MATRIX, SPARSE_GCXS): 0.08768329,
    (NUMPY_MATRIX, SPARSE_COO): 0.0599519,
    (NUMPY_MATRIX, CUPY_SCIPY_CSC): 0.04084738,
    (SCIPY_COO, SCIPY_CSC): 0.01241523,
    (SCIPY_COO, SPARSE_DOK): 0.68932816,
    (SCIPY_COO, CUPY_SCIPY_COO): 0.02920637,
    (SCIPY_COO, SCIPY_CSR): 0.00376677,
    (SCIPY_COO, CUPY): 0.12038468,
    (SCIPY_COO, CUDA): 0.00611036,
    (SCIPY_COO, NUMPY): 0.00636043,
    (SCIPY_COO, CUPY_SCIPY_CSR): 0.01180241,
    (SCIPY_COO, NUMPY_MATRIX): 0.01291071,
    (SCIPY_COO, SCIPY_COO): 6.6e-07,
    (SCIPY_COO, SPARSE_GCXS): 0.00395217,
    (SCIPY_COO, SPARSE_COO): 0.00128454,
    (SCIPY_COO, CUPY_SCIPY_CSC): 0.01739878,
    (SPARSE_GCXS, SCIPY_CSC): 0.38719814,
    (SPARSE_GCXS, SPARSE_DOK): 0.80167217,
    (SPARSE_GCXS, CUPY_SCIPY_COO): 0.52250334,
    (SPARSE_GCXS, SCIPY_CSR): 0.37709386,
    (SPARSE_GCXS, CUPY): 0.05500747,
    (SPARSE_GCXS, CUDA): 0.05242294,
    (SPARSE_GCXS, NUMPY): 0.05339986,
    (SPARSE_GCXS, CUPY_SCIPY_CSR): 0.48872876,
    (SPARSE_GCXS, NUMPY_MATRIX): 0.05962629,
    (SPARSE_GCXS, SCIPY_COO): 0.3914773,
    (SPARSE_GCXS, SPARSE_GCXS): 7.1e-07,
    (SPARSE_GCXS, SPARSE_COO): 0.03642134,
    (SPARSE_GCXS, CUPY_SCIPY_CSC): 0.48377229,
    (SPARSE_COO, SCIPY_CSC): 0.0324796,
    (SPARSE_COO, SPARSE_DOK): 0.73635922,
    (SPARSE_COO, CUPY_SCIPY_COO): 0.05414041,
    (SPARSE_COO, SCIPY_CSR): 0.02293617,
    (SPARSE_COO, CUPY): 0.01775059,
    (SPARSE_COO, CUDA): 0.0172146,
    (SPARSE_COO, NUMPY): 0.01501184,
    (SPARSE_COO, CUPY_SCIPY_CSR): 0.03100931,
    (SPARSE_COO, NUMPY_MATRIX): 0.02309117,
    (SPARSE_COO, SCIPY_COO): 0.02206325,
    (SPARSE_COO, SPARSE_GCXS): 0.03331701,
    (SPARSE_COO, SPARSE_COO): 7.2e-07,
    (SPARSE_COO, CUPY_SCIPY_CSC): 0.03927987,
    (CUPY_SCIPY_CSC, SCIPY_CSC): 0.01082002,
    (CUPY_SCIPY_CSC, SPARSE_DOK): 0.81606757,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_COO): 0.02791661,
    (CUPY_SCIPY_CSC, SCIPY_CSR): 0.00620633,
    (CUPY_SCIPY_CSC, CUPY): 0.07202113,
    (CUPY_SCIPY_CSC, CUDA): 0.01021739,
    (CUPY_SCIPY_CSC, NUMPY): 0.01122473,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_CSR): 0.0032001,
    (CUPY_SCIPY_CSC, NUMPY_MATRIX): 0.06228253,
    (CUPY_SCIPY_CSC, SCIPY_COO): 0.0149932,
    (CUPY_SCIPY_CSC, SPARSE_GCXS): 0.00343712,
    (CUPY_SCIPY_CSC, SPARSE_COO): 0.12067594,
    (CUPY_SCIPY_CSC, CUPY_SCIPY_CSC): 1.41e-06,
}


# In order to support subclasses we cache dynamically
# which type maps to which format code
_type_cache = {}


def get_backend(arr):
    t = type(arr)
    backend = _type_cache.get(t, False)
    if backend is False:
        backend = None
        # Make sure to check NumPy matrix first since numpy.matrix is a subclass
        # of numpy.ndarray
        for b in (NUMPY_MATRIX, ) + tuple(BACKENDS):
            # Always return NUMPY for np.ndarray
            if b == CUDA:
                continue
            try:
                cls = classes[b]
            except (ImportError, ModuleNotFoundError):
                # probably no CuPy
                continue
            if isinstance(arr, cls):
                backend = b
                break
        _type_cache[t] = backend
    return backend


def get_converter(source_backend, target_backend, strict=False):
    identifier = (source_backend, target_backend)
    if strict:
        return converters[identifier]
    else:
        return converters.get(identifier, _identity)


def cheapest_pair(source_backends, target_backends):
    keys = itertools.product(source_backends, target_backends)
    s = sorted(keys, key=lambda x: cost[x])
    cheapest_key = s[0]
    return cheapest_key


def get_cheapest_converter(source_backends, target_backend):
    cheapest_key = cheapest_pair(source_backends, (target_backend, ))
    return cheapest_key[0], converters[cheapest_key]


def for_backend(arr, backend, strict=True):
    source_backend = get_backend(arr)
    converter = get_converter(source_backend, backend, strict)
    # print(source_backend, backend, converter)
    return converter(arr)


def check_shape(arr, shape):
    shape = tuple(shape)
    backend = get_backend(arr)
    if backend in D2_BACKENDS:
        sigprod = prod(shape[1:])
        expected = (shape[0], sigprod)
        if arr.shape != expected:
            raise ValueError(
                f"Mismatching shape {arr.shape} vs expected {expected} for {shape}, "
                f"backend {backend}."
            )
    # Also catches None for an unknown backend
    else:
        if arr.shape != shape:
            raise ValueError(f"Mismatching shape {arr.shape} vs {shape}, backend {backend}.")
    return True


def get_device_class(backend: Optional[ArrayBackend]):
    if backend == CUDA:
        return 'cuda'
    elif backend in CUPY_BACKENDS:
        return 'cupy'
    elif backend in CPU_BACKENDS or backend is None:
        return 'cpu'
    else:
        raise ValueError(f"Unknown backend {backend}.")


def make_like(arr, target, strict=True):
    '''
    Convert to compatible format and shape for assignment into a target array.

    The result of array operations on a sparse input array can't always be
    merged into a result array of different format directly since arrays from
    the :mod:`sparse` package are not converted to NumPy arrays automatically
    and CUDA arrays may have to be transferred to CPU.

    This doesn't support broadcasting, i.e. array and target shape must match,
    minus flattened dimensions for conversion from 2D to nD array formats.
    '''
    check_shape(arr, target.shape)
    res = for_backend(arr, get_backend(target), strict=strict).reshape(target.shape)
    return res


def benchmark_conversions(shape, dtype, density, backends=BACKENDS, repeats=10, warmup=True):
    results = {}

    def data_rvs(size):
        kind = np.dtype(dtype).kind
        if kind == 'b':
            return np.random.choice([True, False], size=size)
        elif kind == 'i':
            return np.random.randint(-128, 128, size=size, dtype=dtype)
        elif kind == 'u':
            return np.random.randint(0, 256, size=size, dtype=dtype)
        elif kind == 'f':
            return np.random.random(size).astype(dtype)
        elif kind == 'c':
            return np.random.random(size).astype(dtype) + 1j*np.random.random(size).astype(dtype)
        else:
            raise ValueError(f"Can't generate data of dtype {dtype}.")

    feedstock = sparse.random(shape=shape, density=density, data_rvs=data_rvs)
    assert feedstock.dtype == dtype

    for left in backends:
        source = for_backend(feedstock, left)
        for right in backends:
            if warmup:
                for_backend(source, right)
            min_time = np.inf
            max_time = -np.inf
            total_time = 0
            for repeat in range(repeats):
                start = time.perf_counter_ns()
                for_backend(source, right)
                stop = time.perf_counter_ns()
                duration = stop - start
                min_time = min(min_time, duration)
                max_time = max(max_time, duration)
                total_time += duration
            res = (min_time / 1e9, max_time / 1e9, total_time / repeats / 1e9)
            results[(left, right)] = res
            print(left, right, res)

    return results
