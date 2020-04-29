import numpy as np

from libertem.udf.sum import SumUDF
from libertem.io.dataset.roi import RoiDataSet


def test_smoke(default_raw, lt_ctx):
    roi = np.zeros(default_raw.shape.nav, dtype=bool)
    roi[0, 0] = True
    rois = [roi]

    roi_ds = RoiDataSet(wrapped=default_raw, rois=rois)

    lt_ctx.run_udf(dataset=roi_ds, udf=SumUDF())
