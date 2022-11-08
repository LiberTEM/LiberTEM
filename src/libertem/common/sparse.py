import numpy as np
import scipy.sparse as sp
import sparse


def to_dense(a):
    if isinstance(a, sparse.SparseArray):
        return a.todense()
    elif sp.issparse(a):
        return a.toarray()
    else:
        return np.array(a)


def to_sparse(a):
    if isinstance(a, sparse.COO):
        return a
    elif isinstance(a, sparse.SparseArray):
        return sparse.COO(a)
    elif sp.issparse(a):
        return sparse.COO.from_scipy_sparse(a)
    else:
        return sparse.COO.from_numpy(np.array(a))


def is_sparse(a):
    return isinstance(a, sparse.SparseArray) or sp.issparse(a)


def assert_sparse(a) -> sparse.SparseArray:
    assert is_sparse(a)
    return a
