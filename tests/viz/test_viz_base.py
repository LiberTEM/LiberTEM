import numpy as np
import pytest
from unittest.mock import Mock
import math

from libertem import viz
from libertem.viz.base import Dummy2DPlot
from libertem.udf.base import NoOpUDF
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF


def test_get_stat_limits_outlier():
    # Array with one outlier
    arr = np.ones(1000)
    arr[500] = 1e6  # extreme outlier
    vmin, vmax = viz.base._get_stat_limits(arr)
    # vmin/vmax should be close to 1, not affected by the outlier
    assert abs(vmin - 1) < 1e-6
    assert abs(vmax - 1) < 1e-2  # allow for quantile rounding


def test_get_stat_limits_all_inf():
    arr = np.array([np.inf, -np.inf])
    vmin, vmax = viz.base._get_stat_limits(arr)
    assert vmin == 1.0
    assert math.isclose(vmax, math.nextafter(1.0, math.inf))


def test_get_stat_limits_constant():
    arr = np.full(100, 42.0)
    vmin, vmax = viz.base._get_stat_limits(arr)
    assert vmin == 42.0
    assert math.isclose(vmax, math.nextafter(42.0, math.inf))


def test_get_stat_limits_no_quantile():
    arr = np.array([1, 2, 3, 100])
    vmin, vmax = viz.base._get_stat_limits(arr, quantile=0)
    assert vmin == 1
    assert vmax == 100


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
