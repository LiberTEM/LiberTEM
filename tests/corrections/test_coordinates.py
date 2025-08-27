import numpy as np
from numpy.testing import assert_allclose

import libertem.corrections.coordinates as c
from libertem.utils import rotate_deg


def test_scale():
    y, x = np.random.random((2, 7))
    factor = np.pi
    r_y, r_x = c.scale(factor) @ (y, x)
    assert np.allclose(r_y, y * factor)
    assert np.allclose(r_x, x * factor)


def test_rotate():
    y, x = np.random.random((2, 7))
    radians = np.pi/180*23
    r_y, r_x = c.identity() @ c.rotate(radians) @ (y, x)
    r_y2, r_x2 = rotate_deg(y, x, 23)
    assert np.allclose(r_y, r_y2)
    assert np.allclose(r_x, r_x2)


def test_rotate_deg():
    y, x = np.random.random((2, 7))
    degrees = 23
    r_y, r_x = c.rotate_deg(degrees) @ (y, x)
    r_y2, r_x2 = rotate_deg(y, x, degrees)
    assert np.allclose(r_y, r_y2)
    assert np.allclose(r_x, r_x2)


def test_flip_y():
    y, x = np.random.random((2, 7))
    r_y, r_x = c.flip_y() @ (y, x)
    assert np.allclose(r_y, -y)
    assert np.allclose(r_x, x)


def test_flip_x():
    y, x = np.random.random((2, 7))
    r_y, r_x = c.flip_x() @ (y, x)
    assert np.allclose(r_y, y)
    assert np.allclose(r_x, -x)


def test_chain():
    y = np.array((17., -23., -3., 7.))
    x = np.array((-13., 11., 5., -2.))
    # Reference result: First flip, then rotate
    r_y1, r_x1 = c.flip_y() @ (y, x)
    r_y2, r_x2 = c.rotate_deg(29) @ (r_y1, r_x1)

    # Combined correctly from right to left
    trans_alt = c.rotate_deg(29) @ c.flip_y()
    r_yalt, r_xalt = trans_alt @ (y, x)

    # Combined incorrectly: reverse order
    trans_wrong = c.flip_y() @ c.rotate_deg(29)
    r_ywrong, r_xwrong = trans_wrong @ (y, x)

    assert_allclose(r_y2, r_yalt)
    assert_allclose(r_x2, r_xalt)

    # Counter test: Make sure test is specific
    assert not np.allclose(r_y2, r_ywrong)
    assert not np.allclose(r_x2, r_xwrong)
