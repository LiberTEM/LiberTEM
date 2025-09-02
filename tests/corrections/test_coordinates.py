import numpy as np
from numpy.testing import assert_allclose

import pytest

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


@pytest.mark.parametrize(
    'scale', (-11, 0.023, 1, 17)
)
@pytest.mark.parametrize(
    'rotate', (-23, -2*np.pi, -np.pi, 0, np.pi, 2*np.pi, 0.1234, -0.2345)
)
@pytest.mark.parametrize(
    'flip_y', (False, True)
)
def test_deconstruct(scale, rotate, flip_y):
    do_flip = c.flip_y() if flip_y else c.identity()
    mat = c.scale(scale) @ c.rotate(rotate) @ do_flip
    # A negative scale corresponds to a 180 degree rotation
    if scale < 0:
        rotate += np.pi
        scale = np.abs(scale)

    res_scale, res_rotate, res_flip = c.scale_rotate_flip_y(mat)
    assert_allclose(res_scale, scale)
    # Compare sinus and cosinus of angle instead of angle itself to avoid impact
    # of numerical precision etc at the wrap-around points between negative and
    # positive side
    assert_allclose(np.sin(rotate), np.sin(res_rotate), atol=1e-7)
    assert_allclose(np.cos(rotate), np.cos(res_rotate), atol=1e-7)
    assert res_flip == flip_y


@pytest.mark.parametrize(
    'scale', (-11, 0.023, 1, 17)
)
@pytest.mark.parametrize(
    'flip_y', (False, True)
)
def test_bad_rotate(scale, flip_y):
    bad_rotate = np.array(((np.cos(0.123), np.sin(0.234)), (-np.sin(0.123), np.cos(0.234))))
    do_flip = c.flip_y() if flip_y else c.identity()
    mat = c.scale(scale) @ bad_rotate @ do_flip
    with pytest.raises(ValueError, match='shear'):
        res_scale, res_rotate, res_flip = c.scale_rotate_flip_y(mat)


@pytest.mark.parametrize(
    'rotate', (-23, -2*np.pi, -np.pi, 0, np.pi, 2*np.pi, 0.1234, -0.2345)
)
@pytest.mark.parametrize(
    'flip_y', (False, True)
)
def test_bad_scale(rotate, flip_y):
    bad_scale = np.array(((2, 0), (0, 3)))
    do_flip = c.flip_y() if flip_y else c.identity()
    mat = bad_scale @ c.rotate(rotate) @ do_flip
    with pytest.raises(ValueError, match='scale'):
        res_scale, res_rotate, res_flip = c.scale_rotate_flip_y(mat)
