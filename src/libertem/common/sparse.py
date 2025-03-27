import numpy as np
import scipy.sparse as sp
import sparse
from typing import Union, TYPE_CHECKING, Optional
import sparseconverter


if TYPE_CHECKING:
    from libertem.common.shape import Shape


def to_dense(a):
    # If unsupported by sparseconverter
    if sparseconverter.get_backend(a) is None:
        return np.array(a)
    else:
        return sparseconverter.for_backend(a, sparseconverter.NUMPY)


def to_sparse(a, shape: Optional[Union['Shape', tuple[int, ...]]] = None):
    if isinstance(a, (tuple, list)):
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
        return sparseconverter.for_backend(a, sparseconverter.SPARSE_COO)


def sparse_to_coo(a, shape: Optional[Union['Shape', tuple[int, ...]]] = None):
    if a is None or isinstance(a, np.ndarray):
        return a
    return to_sparse(a, shape=shape)


def is_sparse(a):
    return isinstance(a, sparse.SparseArray) or sp.issparse(a)


def assert_sparse(a) -> sparse.SparseArray:
    assert is_sparse(a)
    return a
