from typing import Iterable, Union, Tuple

import numpy as np

from .sparse import assert_sparse


_prod_accepted = (
    int, bool,
    np.bool_, np.signedinteger, np.unsignedinteger
)

ProdAccepted = Union[
    int, bool,
    np.bool_, np.signedinteger, np.unsignedinteger
]


def prod(iterable: Iterable[ProdAccepted]):
    '''
    Safe product for large integer size calculations.

    :meth:`numpy.prod` uses 32 bit for default :code:`int` on Windows 64 bit. This
    function uses infinite width integers to calculate the product and
    throws a ValueError if it encounters types other than the supported ones.
    '''
    result = 1

    for item in iterable:
        if isinstance(item, _prod_accepted):
            result *= int(item)
        else:
            raise ValueError()
    return result


def count_nonzero(array) -> int:
    nnz: int  # for mypy
    try:
        nnz = np.count_nonzero(array)
    except TypeError:
        sparse_a = assert_sparse(array)
        nnz = sparse_a.nnz
    return nnz


def flat_nonzero(array):
    return array.flatten().nonzero()[0]


def make_2D_square(shape: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Convert the 1D shape tuple into a square 2D shape tuple
    if the size of shape is a square number

    Non-square dim and len(shape) != 1 are passed directly through.
    shape (1,) is considered square and returns (1, 1)

    Parameters
    ----------
    shape : Tuple[int, ...]
        1D shape tuple[int]

    Returns
    -------
    Tuple[int, ...]
        2D shape tuple[int, int] if shape is 1D and contains a square
        number of elements, else the input shape is returned
    """
    if len(shape) != 1:
        return shape
    size = prod(shape)
    if size < 1:
        raise ValueError('Zero or negative shape.size')
    dim, remainder = divmod(np.sqrt(size), 1)
    if remainder == 0:
        return (dim,) * 2
    return shape


def _sparse_ndenumerate(array):
    flat_array = array.flatten()
    nonzero = flat_nonzero(flat_array)
    for idx in nonzero:
        coords = np.unravel_index(idx, array.shape)
        yield coords, flat_array[idx]


def ndenumerate(array):
    try:
        yield from np.ndenumerate(array)
    except RuntimeError:
        yield from _sparse_ndenumerate(array)
