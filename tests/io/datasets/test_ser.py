import os

import pytest
import numpy as np

from libertem.udf.sum import SumUDF
from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape

SER_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'default.ser')
HAVE_SER_TESTDATA = os.path.exists(SER_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_SER_TESTDATA, reason="need SER testdata")


def test_smoke(lt_ctx):
    ds = lt_ctx.load("ser", path=SER_TESTDATA_PATH)
    p = next(ds.get_partitions())
    tileshape = Shape(
        (1,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    next(p.get_tiles(tiling_scheme))


def test_roi(lt_ctx):
    ds = lt_ctx.load("ser", path=SER_TESTDATA_PATH)
    roi = np.zeros(ds.shape.nav, dtype=np.bool)
    roi[0, 1] = True

    parts = ds.get_partitions()

    p = next(parts)
    tileshape = Shape(
        (1,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    t0 = next(p.get_tiles(tiling_scheme, roi=roi))
    assert t0.tile_slice.origin[0] == 0

    p1 = next(parts)
    roi.reshape((-1,))[p1.slice.get(nav_only=True)] = True

    t1 = next(p1.get_tiles(tiling_scheme, roi=roi))
    assert t1.tile_slice.origin[0] == 1


def test_with_udf(lt_ctx):
    ds = lt_ctx.load("ser", path=SER_TESTDATA_PATH)
    udf = SumUDF()
    udf_results = lt_ctx.run_udf(udf=udf, dataset=ds)

    # compare result with a known-working variant:
    result = np.zeros((ds.shape.sig))
    for p in ds.get_partitions():
        tileshape = Shape(
            (1,) + tuple(ds.shape.sig),
            sig_dims=ds.shape.sig.dims
        )
        tiling_scheme = TilingScheme.make_for_shape(
            tileshape=tileshape,
            dataset_shape=ds.shape,
        )
        for tile in p.get_tiles(tiling_scheme):
            result[tile.tile_slice.get(sig_only=True)] += np.sum(tile.data, axis=0)

    assert np.allclose(udf_results['intensity'].data, result)
