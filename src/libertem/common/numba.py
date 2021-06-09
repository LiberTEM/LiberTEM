import time
import hashlib
import logging

import numba
from numba.core.registry import dispatcher_registry, CPUDispatcher
from numba.core.caching import FunctionCache
from numba.core.serialize import dumps
import numpy as np
import scipy.sparse


logger = logging.getLogger(__name__)


@numba.njit(boundscheck=True)
def numba_ravel_multi_index_single(multi_index, dims):
    # only supports the "single index" case
    idxs = range(len(dims) - 1, -1, -1)
    res = 0
    for idx in idxs:
        stride = 1
        for dimidx in range(idx + 1, len(dims)):
            stride *= dims[dimidx]
        res += multi_index[idx] * stride
    return res


@numba.njit(boundscheck=True)
def numba_ravel_multi_index_multi(multi_index, dims):
    # only supports the "multi index" case
    idxs = range(len(dims) - 1, -1, -1)
    res = np.zeros(len(multi_index[0]), dtype=np.intp)
    for i in range(len(res)):
        for idx in idxs:
            stride = 1
            for dimidx in range(idx + 1, len(dims)):
                stride *= dims[dimidx]
            res[i] += multi_index[idx, i] * stride
    return res


@numba.njit(boundscheck=True)
def numba_unravel_index_single(index, shape):
    sizes = np.zeros(len(shape), dtype=np.intp)
    result = np.zeros(len(shape), dtype=np.intp)
    sizes[-1] = 1
    for i in range(len(shape) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * shape[i + 1]
    remainder = index
    for i in range(len(shape)):
        result[i] = remainder // sizes[i]
        remainder %= sizes[i]
    return result


@numba.njit(boundscheck=True)
def numba_unravel_index_multi(indices, shape):
    sizes = np.zeros(len(shape), dtype=np.intp)
    result = np.zeros((len(shape), len(indices)), dtype=np.intp)
    sizes[-1] = 1
    for i in range(len(shape) - 2, -1, -1):
        sizes[i] = sizes[i + 1] * shape[i + 1]
    remainder = indices
    for i in range(len(shape)):
        result[i] = remainder // sizes[i]
        remainder %= sizes[i]
    return result


def rmatmul(left_dense, right_sparse):
    '''
    Custom implementations for dense-sparse matrix product

    Currently the implementation of __rmatmul__ in scipy.sparse
    uses transposes and the left hand side matrix product,
    which leads to poor performance for large tiles.

    See https://github.com/scipy/scipy/issues/13211
    See https://github.com/LiberTEM/LiberTEM/issues/917

    Parameters
    ----------

    left_dense : numpy.ndarray
        2D left-hand matrix (dense)

    right_sparse : Union[scipy.sparse.csc_matrix, scipy.sparse.csr_matrix]
        2D right hand sparse matrix

    Returns
    -------
    numpy.ndarray : Result of matrix product

    '''
    if len(left_dense.shape) != 2:
        raise ValueError(f"Shape of left_dense is not 2D, but {left_dense.shape}.")
    if len(right_sparse.shape) != 2:
        raise ValueError(f"Shape of right_sparse is not 2D, but {right_sparse.shape}.")
    if left_dense.shape[1] != right_sparse.shape[0]:
        raise ValueError(
            f"Shape mismatch: left_dense.shape[1] != right_sparse.shape[0], "
            f"got {left_dense.shape[1], right_sparse.shape[0]}."
        )
    result_t = np.zeros(
        shape=(right_sparse.shape[1], left_dense.shape[0]),
        dtype=np.result_type(right_sparse, left_dense)
    )

    if isinstance(right_sparse, scipy.sparse.csc_matrix):
        _rmatmul_csc(
            left_dense=left_dense,
            right_data=right_sparse.data,
            right_indices=right_sparse.indices,
            right_indptr=right_sparse.indptr,
            res_inout_t=result_t
        )
    elif isinstance(right_sparse, scipy.sparse.csr_matrix):
        _rmatmul_csr(
            left_dense=left_dense,
            right_data=right_sparse.data,
            right_indices=right_sparse.indices,
            right_indptr=right_sparse.indptr,
            res_inout_t=result_t
        )
    else:
        raise ValueError(
            f"Right hand matrix mus be of type scipy.sparse.csc_matrix or scipy.sparse.csr_matrix, "
            f"got {type(right_sparse)}."
        )
    return result_t.T.copy()


@numba.njit(fastmath=True, cache=True)
def _rmatmul_csc(left_dense, right_data, right_indices, right_indptr, res_inout_t):
    left_rows = left_dense.shape[0]
    for right_column in range(len(right_indptr) - 1):
        offset = right_indptr[right_column]
        items = right_indptr[right_column+1] - offset
        if items > 0:
            for i in range(items):
                index = i + offset
                right_row = right_indices[index]
                right_value = right_data[index]
                for left_row in range(left_rows):
                    tmp = left_dense[left_row, right_row] * right_value
                    res_inout_t[right_column, left_row] += tmp


@numba.njit(fastmath=True, cache=True)
def _rmatmul_csr(left_dense, right_data, right_indices, right_indptr, res_inout_t):
    left_rows = left_dense.shape[0]
    rowbuf = np.empty(shape=(left_rows,), dtype=left_dense.dtype)
    for right_row in range(len(right_indptr) - 1):
        offset = right_indptr[right_row]
        items = right_indptr[right_row+1] - offset
        if items > 0:
            rowbuf[:] = left_dense[:, right_row]
            for i in range(items):
                index = i + offset
                right_column = right_indices[index]
                right_value = right_data[index]
                for left_row in range(left_rows):
                    tmp = rowbuf[left_row] * right_value
                    res_inout_t[right_column, left_row] += tmp


_cached_njit_reg = []


def cached_njit(*args, **kwargs):
    """
    Replacement for numba.njit with custom caching. Only supports usage
    with parameters, i.e.

    @cached_njit()
    def fn():
        ...
    """
    def wrapper(fn):
        kwargs.update({'_target': 'custom_cpu', 'cache': True})
        _cached_njit_reg.append(fn)
        return numba.njit(fn, *args, **kwargs)
    return wrapper


def hasher(x):
    return hashlib.sha256(x).hexdigest()


class MyFunctionCache(FunctionCache):
    def _get_dependencies(self, cvar):
        deps = [cvar]
        if hasattr(cvar, 'py_func'):
            # TODO: does the cache key need to depend on any other
            # attributes of the Dispatcher?
            closure = cvar.py_func.__closure__
            deps = [cvar.py_func.__code__.co_code]
        elif hasattr(cvar, '__closure__'):
            closure = cvar.__closure__
            # if cvar is a function and closes over a Dispatcher, the
            # cache will be busted because of the uuid that is regenerated
            deps = [cvar.__code__.co_code]
        else:
            closure = None
        if closure is not None:
            for x in closure:
                deps.extend(self._get_dependencies(x.cell_contents))
        return deps

    def _index_key(self, sig, codegen):
        """
        Compute index key for the given signature and codegen.
        It includes a description of the OS, target architecture and hashes of
        the bytecode for the function and, if the function has a __closure__,
        a hash of the cell_contents.
        """
        codebytes = self._py_func.__code__.co_code
        cvars = self._get_dependencies(self._py_func)
        if len(cvars) > 0:
            cvarbytes = dumps(cvars)
        else:
            cvarbytes = b''

        return (sig, "libertem-numba-cache", codegen.magic_tuple(), (hasher(codebytes),
                                             hasher(cvarbytes),))

    def load_overload(self, *args, **kwargs):
        t0 = time.time()
        data = super().load_overload(*args, **kwargs)
        t1 = time.time()
        if data is None:
            logger.info("numba cache miss %s %s" % (self._name, self._py_func))
        else:
            logger.info("cache hit for %s, load took %.3fs" % (self._name, (t1 - t0)))
        return data


class MyCPUDispatcher(CPUDispatcher):
    def enable_caching(self):
        self._cache = MyFunctionCache(self.py_func)


dispatcher_registry['custom_cpu'] = MyCPUDispatcher
# dispatcher_registry['custom_cpu'] = CPUDispatcher


def prime_numba_cache(ds):
    dtypes = (np.float32, None)
    for dtype in dtypes:
        roi = np.zeros(ds.shape.nav, dtype=bool).reshape((-1,))
        roi[max(-ds._meta.sync_offset, 0)] = True

        from libertem.udf.sum import SumUDF
        from libertem.udf.raw import PickUDF
        from libertem.corrections.corrset import CorrectionSet
        from libertem.io.dataset.base import Negotiator

        # need to have at least one UDF; here we run for both sum and pick
        # to reduce the initial latency when switching to pick mode
        udfs = [SumUDF(), PickUDF()]
        neg = Negotiator()
        for udf in udfs:
            for corr_dtype in (np.float32, None):
                if corr_dtype is not None:
                    corrections = CorrectionSet(dark=np.zeros(ds.shape.sig, dtype=corr_dtype))
                else:
                    corrections = None
                found_first_tile = False
                for p in ds.get_partitions():
                    if found_first_tile:
                        break
                    p.set_corrections(corrections)
                    tiling_scheme = neg.get_scheme(
                        udfs=[udf],
                        partition=p,
                        read_dtype=dtype,
                        roi=roi,
                        corrections=corrections,
                    )
                    for t in p.get_tiles(tiling_scheme=tiling_scheme, roi=roi):
                        found_first_tile = True
                        break
