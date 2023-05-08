import logging

import numba
import scipy.sparse
import numpy as np

from .cache import cached_njit  # NOQA: F401

logger = logging.getLogger(__name__)

numba_dtypes = frozenset(np.dtype(t) for t in (
    bool,
    int,
    float,
    np.bool_,
    np.uint8,
    np.int8,
    np.uint16,
    np.int16,
    np.uint32,
    np.int32,
    np.uint64,
    np.int64,
    np.float32,
    np.float64,
    np.complex64,
    np.complex128,
    np.intc,
    np.intp,
    np.uintc,
    np.uintp
))


@numba.njit(boundscheck=True, nogil=True)
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


@numba.njit(boundscheck=True, nogil=True)
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


@numba.njit(boundscheck=True, nogil=True)
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


@numba.njit(boundscheck=True, nogil=True)
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


@numba.njit(fastmath=True, cache=True, nogil=True)
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


@numba.njit(fastmath=True, cache=True, nogil=True)
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
