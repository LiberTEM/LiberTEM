import numba
import numpy as np


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
