import numpy as np
import pytest
from unittest.mock import Mock

from libertem import viz
from libertem.viz.base import Dummy2DPlot
from libertem.udf.base import NoOpUDF
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF


def test_rgb_from_vector():
    rgb = viz.CMAP_CIRCULAR_DEFAULT.rgb_from_vector((0, 0, 0))  # center (grey)
    np.testing.assert_equal(rgb, np.asarray([127, 127, 127], dtype=np.uint8))
    rgb = viz.CMAP_CIRCULAR_DEFAULT.rgb_from_vector((0, 1, 0))  # up (green)
    np.testing.assert_equal(rgb, np.asarray([78, 173, 19], dtype=np.uint8))
    rgb = viz.CMAP_CIRCULAR_DEFAULT.rgb_from_vector((1, 0, 0))  # right (red)
    np.testing.assert_equal(rgb, np.asarray([230, 87, 64], dtype=np.uint8))
    rgb = viz.CMAP_CIRCULAR_DEFAULT.rgb_from_vector((0, -1, 0))  # down (purple)
    np.testing.assert_equal(rgb, np.asarray([177, 81, 234], dtype=np.uint8))
    rgb = viz.CMAP_CIRCULAR_DEFAULT.rgb_from_vector((-1, 0, 0))  # left (cyan)
    np.testing.assert_equal(rgb, np.asarray([24, 167, 191], dtype=np.uint8))


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
        plot = viz.CMAP_CIRCULAR_DEFAULT.rgb_from_vector((data, 0, 0))
        return (plot, damage)

    plots = [Dummy2DPlot(dataset=default_raw, udf=udf, channel=RGB_plot)]
    lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=plots)

    assert plots[0].data.shape == tuple(default_raw.shape.nav) + (3, )
