import numpy as np

from typing import Union, TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from libertem.common.shape import Shape

from .array_backends import get_backend, for_backend, NUMPY, SPARSE_COO, SPARSE_BACKENDS


def to_dense(a):
    res = for_backend(a, NUMPY)
    if res.flags.c_contiguous:
        return res
    else:
        return np.array(res)


def to_sparse(a):
    return for_backend(a, SPARSE_COO)


def sparse_to_coo(a, shape: Optional[Union['Shape', Tuple[int]]] = None):
    if a is None or isinstance(a, np.ndarray):
        return a
    return to_sparse(a, shape=shape)


def is_sparse(a):
    return get_backend(a) in SPARSE_BACKENDS
