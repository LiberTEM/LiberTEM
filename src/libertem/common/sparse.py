import numpy as np
<<<<<<< HEAD
import scipy.sparse as sp
import sparse
from typing import Union, TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from libertem.common.shape import Shape
=======

from .array_formats import array_format, as_format, NUMPY, SPARSE_COO, SPARSEFORMATS
>>>>>>> a8c5e412 (WIP intermediate save)


def to_dense(a):
    res = as_format(a, NUMPY)
    if res.flags.c_contiguous:
        return res
    else:
        return np.array(res)


def to_sparse(a):
    return as_format(a, SPARSE_COO)


def sparse_to_coo(a, shape: Optional[Union['Shape', Tuple[int]]] = None):
    if a is None or isinstance(a, np.ndarray):
        return a
    return to_sparse(a, shape=shape)


def is_sparse(a):
    return array_format(a) in SPARSEFORMATS
