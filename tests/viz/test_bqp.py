from unittest import mock

import IPython

from libertem.viz.bqp import BQLive2DPlot
from libertem.udf.sum import SumUDF


def test_bqp_smoke(monkeypatch, lt_ctx, default_raw):
    udf = SumUDF()
    monkeypatch.setattr(IPython.display, 'display', mock.MagicMock())
    monkeypatch.setattr(lt_ctx, 'plot_class', BQLive2DPlot)
    lt_ctx.run_udf(dataset=default_raw, udf=udf, plots=True)
    IPython.display.display.assert_called()
