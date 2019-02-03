import numpy as np

import libertem.analysis.karina1_udf as karina


def test_refinenemt():
    data = np.array([
        (0, 0, 0, 0, 0, 1),
        (0, 1, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (2, 3, 0, 0, 0, 0),
        (0, 2, 0, 0, 0, -10)
    ])

    assert np.allclose(karina.refine_center(center=(1, 1), r=1, corrmap=data), (1, 1))
    assert np.allclose(karina.refine_center(center=(2, 2), r=1, corrmap=data), (1, 1))
    assert np.allclose(karina.refine_center(center=(1, 4), r=1, corrmap=data), (0.5, 4.5))

    y, x = (4, 1)
    ry, rx = karina.refine_center(center=(y, x), r=1, corrmap=data)
    assert (ry > y) and (ry < (y + 1))
    assert (rx < x) and (rx > (x - 1))

    y, x = (4, 4)
    ry, rx = karina.refine_center(center=(y, x), r=1, corrmap=data)
    assert (ry < y) and (ry > (y - 1))
    assert (rx < x) and (rx > (x - 1))
