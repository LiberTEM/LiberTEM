from unittest import mock

import numpy as np
import pytest
import matplotlib.pyplot

from libertem.viz.mpl import MPLLive2DPlot
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF


def test_mpl_smoke(monkeypatch, lt_ctx, default_raw):
    udf = SumUDF()
    monkeypatch.setattr(
        matplotlib.pyplot,
        'subplots',
        mock.MagicMock(return_value=(mock.MagicMock(), mock.MagicMock()))
    )
    monkeypatch.setattr(lt_ctx, 'plot_class', MPLLive2DPlot)
    lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=True)
    matplotlib.pyplot.subplots.assert_called()


def test_mpl_nodisplay(monkeypatch, lt_ctx, default_raw):
    udf = SumUDF()
    monkeypatch.setattr(
        matplotlib.pyplot,
        'subplots',
        mock.MagicMock(return_value=(mock.MagicMock(), mock.MagicMock()))
    )
    plots = [MPLLive2DPlot(dataset=default_raw, udf=udf)]
    with pytest.warns(UserWarning, match="Plot is not displayed, not plotting."):
        lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=plots)


def test_empty(monkeypatch, lt_ctx, default_raw):
    udf = SumSigUDF()
    monkeypatch.setattr(
        matplotlib.pyplot,
        'subplots',
        mock.MagicMock(return_value=(mock.MagicMock(), mock.MagicMock()))
    )
    monkeypatch.setattr(lt_ctx, 'plot_class', MPLLive2DPlot)

    def extract(udf_result, damage):
        return udf_result['intensity'].data, np.zeros_like(damage)

    plot = MPLLive2DPlot(dataset=default_raw, udf=udf, channel=extract)

    lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=[plot])
