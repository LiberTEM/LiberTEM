import os

import pytest

from libertem.io.dataset.seq import SEQDataSet
from libertem.common import Shape
from libertem.io.dataset.base import TilingScheme

SEQ_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'default.seq')
HAVE_SEQ_TESTDATA = os.path.exists(SEQ_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_SEQ_TESTDATA, reason="need .seq testdata")


def get_first_tile(p0, tiling_scheme):
    return next(p0.get_tiles(tiling_scheme))


def test_positive_sync_offset_seq(lt_ctx, benchmark):
    nav_shape = (8, 8)
    sync_offset = 2

    ds = SEQDataSet(
        path=SEQ_TESTDATA_PATH, nav_shape=nav_shape, sync_offset=sync_offset
    )
    ds.set_num_cores(4)
    ds = ds.initialize(lt_ctx.executor)

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    p0 = next(ds.get_partitions())
    benchmark(get_first_tile, p0=p0, tiling_scheme=tiling_scheme)


def test_negative_sync_offset_seq(lt_ctx, benchmark):
    nav_shape = (8, 8)
    sync_offset = -2

    ds = SEQDataSet(
        path=SEQ_TESTDATA_PATH, nav_shape=nav_shape, sync_offset=sync_offset
    )
    ds.set_num_cores(4)
    ds = ds.initialize(lt_ctx.executor)

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    p0 = next(ds.get_partitions())
    benchmark(get_first_tile, p0=p0, tiling_scheme=tiling_scheme)


def test_no_sync_offset_seq(lt_ctx, benchmark):
    nav_shape = (8, 8)

    ds = SEQDataSet(
        path=SEQ_TESTDATA_PATH, nav_shape=nav_shape
    )
    ds.set_num_cores(4)
    ds = ds.initialize(lt_ctx.executor)

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    p0 = next(ds.get_partitions())
    benchmark(get_first_tile, p0=p0, tiling_scheme=tiling_scheme)
