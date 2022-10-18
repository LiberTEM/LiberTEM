import numpy as np
import pytest

from libertem.common.math import prod, make_2D_square
from libertem.common.shape import Shape


@pytest.mark.parametrize(
    ('sequence', 'ref', 'typ'), [
        ([], 1, int),
        (np.array([]), 1, int),
        ((1, 2, 3), 6, int),
        ((-11, 2, 3), -66, int),
        ((1., 2, False), 0, ValueError),
        (np.array((1., 1+2j, 1), dtype=np.complex128), 1, ValueError),
        ((2**32, 2**32, 2**32), 2**96, int),
        (np.array((2**62, 2**62, 2**62), dtype=np.int64), 2**(3*62), int),
        (Shape((1, 2, 3), sig_dims=1).nav, 2, int)
    ]
)
def test_prod(sequence, ref, typ):
    if typ is int:
        res = prod(sequence)
        assert res == ref
        assert isinstance(res, typ)
    else:
        with pytest.raises(typ):
            res = prod(sequence)


@pytest.mark.parametrize(
    ('shape', 'result'), [
        ((16,), (4, 4)),
        ((100,), (10, 10)),
        ((15,), (15,)),
        ((1,), (1, 1)),
        ((-20,), ValueError),
        ((3, 7), (3, 7)),
        (tuple(), tuple()),
        ((0.4,), ValueError),
    ]
)
def test_make_2D_square(shape, result):
    if isinstance(result, tuple):
        assert make_2D_square(shape) == result
    elif issubclass(result, Exception):
        with pytest.raises(result):
            make_2D_square(shape)
            return
