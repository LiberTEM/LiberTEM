import numpy as np

import libertem.utils as ut


def test_polar():
    data = np.array([
        [(0, 1), (0, 1)],
        [(1, 0), (2, 0)],
        [(-2, 0), (0, 1)],
    ])
    expected = np.array([
        [(1, 0), (1, 0)],
        [(1, np.pi/2), (2, np.pi/2)],
        [(2, -np.pi/2), (1, 0)],
    ])

    result = ut.make_polar(data)
    assert(data.shape == expected.shape)
    assert(result.shape == expected.shape)
    assert(np.allclose(expected, result))


def test_conversion(points):
    assert(np.allclose(points, ut.make_cartesian(ut.make_polar(points))))
