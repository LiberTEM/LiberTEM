from unittest import mock
import sys

import numpy as np
import pytest

from libertem.viz.gms import GMSLive2DPlot
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF


def test_gms_smoke(monkeypatch, lt_ctx, default_raw):
    udf = SumUDF()
    monkeypatch.setitem(sys.modules, 'DigitalMicrograph', mock.MagicMock())
    monkeypatch.setattr(lt_ctx, 'plot_class', GMSLive2DPlot)
    lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=True)
    sys.modules['DigitalMicrograph'].CreateImage.assert_called()


def test_gms_nodisplay(monkeypatch, lt_ctx, default_raw):
    udf = SumUDF()
    monkeypatch.setitem(sys.modules, 'DigitalMicrograph', mock.MagicMock())
    plots = [GMSLive2DPlot(dataset=default_raw, udf=udf)]
    with pytest.warns(UserWarning, match="Plot is not displayed, not plotting."):
        lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=plots)


def test_empty(monkeypatch, lt_ctx, default_raw):
    udf = SumSigUDF()
    monkeypatch.setitem(sys.modules, 'DigitalMicrograph', mock.MagicMock())
    monkeypatch.setattr(lt_ctx, 'plot_class', GMSLive2DPlot)

    def extract(udf_result, damage):
        return udf_result['intensity'].data, np.zeros_like(damage)

    plot = GMSLive2DPlot(dataset=default_raw, udf=udf, channel=extract)

    with pytest.warns(UserWarning, match='^Plot is not displayed, not plotting'):
        lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=[plot])
