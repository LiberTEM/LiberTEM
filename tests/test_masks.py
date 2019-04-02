import numpy as np

import libertem.masks as m


def test_background_substraction():
    mask = m.background_substraction(10, 10, 20, 20, 5, 3)
    assert(np.allclose(np.sum(mask), 0))
