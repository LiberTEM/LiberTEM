import numpy as np

from libertem.common import Shape
from libertem.udf.sum import SumUDF
from libertem.io.dataset.roi import RoiDataSet
from libertem.io.dataset.base import TilingScheme


def test_smoke_udf(default_raw, lt_ctx):
    roi = np.zeros(default_raw.shape.nav, dtype=bool)
    roi[0, 0] = True
    rois = [roi]

    roi_ds = RoiDataSet(wrapped=default_raw, rois=rois)

    lt_ctx.run_udf(dataset=roi_ds, udf=SumUDF())


def test_smoke_simple(default_raw, lt_ctx):
    roi = np.zeros(default_raw.shape.nav, dtype=bool)
    roi[0, 0] = True
    rois = [roi]

    tileshape = Shape(
        (16,) + tuple(default_raw.shape.sig),
        sig_dims=default_raw.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_raw.shape,
    )

    roi_ds = RoiDataSet(wrapped=default_raw, rois=rois)
    for p in roi_ds.get_partitions():
        for tile in p.get_tiles(tiling_scheme=tiling_scheme):
            pass
