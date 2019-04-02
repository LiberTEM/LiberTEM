import numpy as np

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


def test_norm_negative():
    data = -1 * np.ones((16, 16))
    viz.visualize_simple(data)
