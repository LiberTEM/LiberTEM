import numpy as np
import pytest
from unittest import mock

import IPython

from libertem.viz.bqp import BQLive2DPlot
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF


def test_bqp_smoke(monkeypatch, lt_ctx, default_raw):
    pytest.importorskip("bqplot")
    udf = SumUDF()
    monkeypatch.setattr(IPython.display, 'display', mock.MagicMock())
    monkeypatch.setattr(lt_ctx, 'plot_class', BQLive2DPlot)
    lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=True)
    IPython.display.display.assert_called()


def test_bqp_asymm(monkeypatch, lt_ctx, default_raw_asymm):
    pytest.importorskip("bqplot")
    udfs = [SumSigUDF(), SumUDF()]
    monkeypatch.setattr(IPython.display, 'display', mock.MagicMock())
    monkeypatch.setattr(lt_ctx, 'plot_class', BQLive2DPlot)
    lt_ctx.run_udf(dataset=default_raw_asymm, udf=udfs, plots=True)
    IPython.display.display.assert_called()


def test_empty(monkeypatch, lt_ctx, default_raw_asymm):
    default_raw = default_raw_asymm
    pytest.importorskip("bqplot")
    udf = SumSigUDF()
    monkeypatch.setattr(IPython.display, 'display', mock.MagicMock())
    monkeypatch.setattr(lt_ctx, 'plot_class', BQLive2DPlot)

    def extract(udf_result, damage):
        return udf_result['intensity'].data, np.zeros_like(damage)

    plot = BQLive2DPlot(dataset=default_raw, udf=udf, channel=extract)

    lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=[plot])
