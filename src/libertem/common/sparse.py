import numpy as np
import scipy.sparse as sp
import sparse
from typing import Union, TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from libertem.common.shape import Shape


def to_dense(a):
    if isinstance(a, sparse.SparseArray):
        return a.todense()
    elif sp.issparse(a):
        return a.toarray()
    else:
        return np.array(a)


def to_sparse(a, shape: Optional[Union['Shape', Tuple[int]]] = None):
    if isinstance(a, sparse.COO):
        return a
    elif isinstance(a, sparse.SparseArray):
        return sparse.COO(a)
    elif sp.issparse(a):
        return sparse.COO.from_scipy_sparse(a)
    elif isinstance(a, (tuple, list)):
        if all(isinstance(aa, int) for aa in a):
            a = ((a, True),)
        unique = {aa[-1] for aa in a}
        if len(unique) != 1:
            raise ValueError('Cannot cast iterable roi coords with '
                             f'more than one truth value {unique}')
        roi_val = bool(tuple(unique)[0])
        fill_val = not roi_val
        return sparse.COO.from_iter(a, shape=tuple(shape), fill_value=fill_val, dtype=bool)
    else:
        return sparse.COO.from_numpy(np.array(a))


def sparse_to_coo(a, shape: Optional[Union['Shape', Tuple[int]]] = None):
    if a is None or isinstance(a, np.ndarray):
        return a
    return to_sparse(a, shape=shape)


def is_sparse(a):
    return isinstance(a, sparse.SparseArray) or sp.issparse(a)


def assert_sparse(a) -> sparse.SparseArray:
    assert is_sparse(a)
    return a
