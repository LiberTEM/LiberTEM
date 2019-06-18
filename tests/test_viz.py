import numpy as np
import pytest

from libertem import viz


def test_rgb_from_vector():
    rgb = viz.CMAP_CIRCULAR_DEFAULT.rgb_from_vector((0, 0))  # center (grey)
    np.testing.assert_equal(rgb, np.asarray([127, 127, 127], dtype=np.uint8))
    rgb = viz.CMAP_CIRCULAR_DEFAULT.rgb_from_vector((0, 1))  # up (green)
    np.testing.assert_equal(rgb, np.asarray([78, 173, 19], dtype=np.uint8))
    rgb = viz.CMAP_CIRCULAR_DEFAULT.rgb_from_vector((1, 0))  # right (red)
    np.testing.assert_equal(rgb, np.asarray([230, 87, 64], dtype=np.uint8))
    rgb = viz.CMAP_CIRCULAR_DEFAULT.rgb_from_vector((0, -1))  # down (purple)
    np.testing.assert_equal(rgb, np.asarray([177, 81, 234], dtype=np.uint8))
    rgb = viz.CMAP_CIRCULAR_DEFAULT.rgb_from_vector((-1, 0))  # left (cyan)
    np.testing.assert_equal(rgb, np.asarray([24, 167, 191], dtype=np.uint8))


def test_interpolate_color():
    rgb = viz.interpolate_color(0.5, (0, 0, 0), (1, 1, 1))
    np.testing.assert_equal(rgb, (0.5, 0.5, 0.5))


@pytest.mark.parametrize("log", [True, False])
def test_all_nan(log):
    data = np.full((16, 16), np.nan)
    viz.visualize_simple(data, logarithmic=log)


@pytest.mark.parametrize("log", [True, False])
def test_all_ones(log):
    data = np.ones((16, 16))
    viz.visualize_simple(data, logarithmic=log)


@pytest.mark.parametrize("log", [True, False])
def test_all_zeros(log):
    data = np.zeros((16, 16))
    viz.visualize_simple(data, logarithmic=log)


@pytest.mark.parametrize("log", [True, False])
def test_all_negative(log):
    data = np.full((16, 16), -1)
    viz.visualize_simple(data, logarithmic=log)


@pytest.mark.parametrize("log", [True, False])
def test_some_nonnegative(log):
    data = np.full((16, 16), -1)
    data[0] = 0
    data[1] = 1
    viz.visualize_simple(data, logarithmic=log)
