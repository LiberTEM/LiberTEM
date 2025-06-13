import numpy as np
import pytest
from unittest.mock import Mock
import math

from libertem import viz
from libertem.viz.base import Dummy2DPlot
from libertem.udf.base import NoOpUDF
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF


@pytest.mark.parametrize(
    "data,quantile,snip_factor,expected",
    [
        # Empty array
        (np.array([]), 0.001, 5.0, (1.0, math.nextafter(1.0, math.inf))),
        # Single value
        (np.array([3.0]), 0.001, 5.0, (3.0, math.nextafter(3.0, math.inf))),
        # all-inf
        (np.array([np.inf, -np.inf]), 0.001, 5.0, (1.0, math.nextafter(1.0, math.inf))),
        # All zeros
        (np.zeros(10), 0.001, 5.0, (0.0, math.nextafter(0.0, math.inf))),
        # All constant
        (np.full(10, 42.), 0.001, 5.0, (42.0, math.nextafter(42.0, math.inf))),
        # No zeros, quantile=0 (no outlier rejection)
        (np.array([1, 2, 3, 100000]), 0.0, 5.0, (1.0, 100000.0)),
        # Outliers, quantile > 0, snip_factor high (no snipping)
        (np.concatenate(
            [np.ones(100), [1000]]
        ), 0.01, 1000.0, (1.0, pytest.approx(1000., abs=1e-2))),
        # Outliers, quantile > 0, snip_factor low (snipping occurs)
        (np.concatenate([np.ones(100), [1000]]), 0.01, 0.1, (1.0, pytest.approx(1.0, abs=1e-2))),
        # Negative and positive values, zeros present
        (np.array([-5, 0, 5, 10]), 0.25, 5.0, (-5.0, 10.0)),
        # Very sparse data (mostly zeros), aggressive snipping quantile and low
        # snip factor, but snip rejected because zeros are ignored in the
        # statistics and therefore effectively quantile of single value
        (np.concatenate([np.zeros(1000), np.array([5000, ])]), 0.1, 1.0, (0.0, 5000.0)),
        # Snipped because of large difference to next-smaller values compared to snip factor
        (np.concatenate([np.zeros(1000), np.array([1000, 5000, ])]), 0.1, 1.0, (0.0, 1000.0)),
        # Not snipped because of large-enough snip factor
        (np.concatenate([np.zeros(1000), np.array([1000, 5000, ])]), 0.1, 5.0, (0.0, 5000.0)),
        # Snipped because of small snip factor
        (np.concatenate([np.zeros(1000), np.full(30, 1000), [5000]]), 0.001, 1.0, (0.0, 1000.0)),
        # Only last snipped because of inclusive quantile
        (np.concatenate(
            [np.zeros(1000), np.full(30, 1000), [4500, 10000]]
        ), 0.001, 1.0, (0.0, 4500.0)),
        # last two snipped because of exclusive quantile
        (np.concatenate(
            [np.zeros(1000), np.full(30, 1000), [4500, 10000]]
        ), 0.1, 1.0, (0.0, 1000.0)),
        # Negative values, symmetry

        # Very sparse data (mostly zeros), aggressive snipping quantile and low
        # snip factor, but snip rejected because zeros are ignored in the statistics
        (np.concatenate([np.zeros(1000), np.array([-5000, ])]), 0.1, 1.0, (-5000.0, 0.0)),
        # Snipped because of large difference to next-smaller values
        (np.concatenate([np.zeros(1000), np.array([-1000, -5000, ])]), 0.1, 1.0, (-1000.0, 0.0)),
        # Not snipped because of large snip factor
        (np.concatenate([np.zeros(1000), np.array([-1000, -5000, ])]), 0.1, 5.0, (-5000.0, 0.0)),
        # Snipped because of small snip factor
        (np.concatenate([np.zeros(1000), np.full(30, -1000), [-5000]]), 0.001, 1.0, (-1000.0, 0.0)),
        # Mixed positive and negative values, symmetric
        # Snipped because of large difference to next-smaller values
        (np.concatenate(
            [np.zeros(1000), np.array([-1000, -5000, 1000, 5000])]
        ), 0.1, 1.0, (-1000.0, 1000.0)),
        # Not snipped because of large snip factor
        (np.concatenate(
            [np.zeros(1000), np.array([-1000, -5000, 1000, 5000])]
        ), 0.1, 5.0, (-5000.0, 5000.0)),
        # Snipped because of small snip factor
        (np.concatenate([
            np.zeros(1000),
            np.full(30, -2000), [-6000],
            np.full(30, 1000), [5000]
        ]), 0.001, 1.0, (-2000.0, 1000.0)),
        # Partially snipped because of snip factor, asymmetric
        (np.concatenate(
            [np.zeros(1000), np.array([-1000, -5000, 1000, 2000])]
        ), 0.1, 3.0, (-1000.0, 2000.0)),
        (np.array([True, False]), 0.001, 5, (False, True)),
        (np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 255]).astype(np.uint8), 0.001, 5, (0, 255)),
        # No snip uint8
        (
            np.concatenate([np.ones(100), [255]]).astype(np.uint8),
            0.01, 0.1, (1.0, pytest.approx(1.0, abs=1e-2))
        ),
        # Snip uint8
        (
            np.concatenate([np.ones(100), [255]]).astype(np.uint8),
            0.01, 1000, (1.0, pytest.approx(255, abs=1e-2))
        ),
        # Complex numbers are sorted by real part, no snipping since quantile not supported.
        # Return only real part for vmin and vmax.
        (np.array([0-23j, 255+1j]), 0.001, 5, (0, 255)),
    ]
)
def test_get_stat_limits_parametric(data, quantile, snip_factor, expected):
    # Shuffle to ensure order does not affect results
    rng = np.random.default_rng()
    rng.shuffle(data)
    vmin, vmax = viz.base._get_stat_limits(data, quantile=quantile, snip_factor=snip_factor)
    if isinstance(expected[0], float) or isinstance(expected[0], int):
        assert vmin == pytest.approx(expected[0], abs=1e-6)
    else:
        assert vmin == expected[0]
    if isinstance(expected[1], float) or isinstance(expected[1], int):
        assert vmax == pytest.approx(expected[1], abs=1e-6)
    else:
        assert vmax == expected[1]


def test_rgb_from_vector():
    rgb = viz.rgb_from_2dvector(x=0, y=0)  # center (grey)
    np.testing.assert_equal(rgb, np.asarray([127, 127, 127], dtype=np.uint8))

    x = 0
    y = 1
    rgb = viz.rgb_from_2dvector(x=x, y=y)  # up (green)
    # plus green
    np.testing.assert_equal(np.argmax(rgb), 1)
    angle = np.arctan2(y, x)
    mapped = (angle + np.pi) / (2*np.pi)
    cyclic = (np.asarray(viz.libertem_cyclic(mapped)) * 255).astype(np.uint8)
    np.testing.assert_allclose(cyclic[:-1], rgb, atol=2)

    x = 1
    y = 0
    rgb = viz.rgb_from_2dvector(x=x, y=y)  # right (red)
    # plus red
    np.testing.assert_equal(np.argmax(rgb), 0)
    angle = np.arctan2(y, x)
    mapped = (angle + np.pi) / (2*np.pi)
    cyclic = (np.asarray(viz.libertem_cyclic(mapped)) * 255).astype(np.uint8)
    np.testing.assert_allclose(cyclic[:-1], rgb, atol=2)

    x = 0
    y = -1
    rgb = viz.rgb_from_2dvector(x=x, y=y)  # down (purple)
    # minus green, plus blue
    np.testing.assert_equal(np.argmin(rgb), 1)
    np.testing.assert_equal(np.argmax(rgb), 2)
    angle = np.arctan2(y, x)
    mapped = (angle + np.pi) / (2*np.pi)
    cyclic = (np.asarray(viz.libertem_cyclic(mapped)) * 255).astype(np.uint8)
    np.testing.assert_allclose(cyclic[:-1], rgb, atol=2)

    x = -1
    y = 0
    rgb = viz.rgb_from_2dvector(x=x, y=y)  # left (cyan)
    # minus red, plus blue
    np.testing.assert_equal(np.argmin(rgb), 0)
    np.testing.assert_equal(np.argmax(rgb), 2)
    angle = np.arctan2(y, x)
    mapped = (angle + np.pi) / (2*np.pi)
    cyclic = (np.asarray(viz.libertem_cyclic(mapped)) * 255).astype(np.uint8)
    np.testing.assert_allclose(cyclic[:-1], rgb, atol=2)


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


def test_live_nochannels(default_raw):
    udf = NoOpUDF()
    with pytest.raises(ValueError):
        Dummy2DPlot(dataset=default_raw, udf=udf)


def test_live_plotupdate(lt_ctx, default_raw):
    udf = SumUDF()
    m = Mock()
    m.get_udf.return_value = udf
    plots = [m]
    lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=plots)
    n_part = default_raw.get_num_partitions()
    print("Num partitions", n_part)
    print("Mock calls", m.new_data.mock_calls)
    # Otherwise test is meaningless, no intermediate updates
    assert n_part > 1
    # Partition updates plus final update
    assert len(m.new_data.mock_calls) == n_part + 1


@pytest.mark.parametrize('udf_cls', (SumUDF, SumSigUDF))
def test_live_autoextraction(lt_ctx, default_raw, udf_cls):
    udf = udf_cls()
    plots = [Dummy2DPlot(dataset=default_raw, udf=udf, channel=('intensity', np.abs))]
    lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=plots)


def test_live_RGB(lt_ctx, default_raw):
    udf = SumSigUDF()

    def RGB_plot(udf_result, damage):
        data = udf_result['intensity'].data
        plot = viz.rgb_from_2dvector(x=data, y=0)
        return (plot, damage)

    plots = [Dummy2DPlot(dataset=default_raw, udf=udf, channel=RGB_plot)]
    lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=plots)

    assert plots[0].data.shape == tuple(default_raw.shape.nav) + (3, )
