import os

import numpy as np
import pytest

from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.seq import SEQDataSet
from libertem.common import Shape
from libertem.udf.sumsigudf import SumSigUDF
from libertem.io.dataset.base import TilingScheme

from utils import get_testdata_path

SEQ_TESTDATA_PATH = os.path.join(get_testdata_path(), 'default.seq')
HAVE_SEQ_TESTDATA = os.path.exists(SEQ_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_SEQ_TESTDATA, reason="need .seq testdata")


@pytest.fixture
def default_seq(lt_ctx):
    nav_shape = (8, 8)

    ds = SEQDataSet(path=SEQ_TESTDATA_PATH, nav_shape=nav_shape)
    ds.set_num_cores(4)
    ds = ds.initialize(lt_ctx.executor)
    assert tuple(ds.shape) == (8, 8, 128, 128)
    return ds


def test_positive_sync_offset(default_seq, lt_ctx):
    udf = SumSigUDF()
    sync_offset = 2

    ds_with_offset = SEQDataSet(
        path=SEQ_TESTDATA_PATH, nav_shape=(8, 8), sync_offset=sync_offset
    )
    ds_with_offset.set_num_cores(4)
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == 2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (4,) + tuple(ds_with_offset.shape.sig),
        sig_dims=ds_with_offset.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds_with_offset.shape,
    )

    t0 = next(p0.get_tiles(tiling_scheme))
    assert tuple(t0.tile_slice.origin) == (0, 0, 0)

    for p in ds_with_offset.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p.slice.origin == (48, 0, 0)
    assert p.slice.shape[0] == 16

    result = lt_ctx.run_udf(dataset=default_seq, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
        :ds_with_offset._meta.image_count - sync_offset
    ]

    assert np.allclose(result, result_with_offset)


def test_negative_sync_offset(default_seq, lt_ctx):
    udf = SumSigUDF()
    sync_offset = -2

    ds_with_offset = SEQDataSet(
        path=SEQ_TESTDATA_PATH, nav_shape=(8, 8), sync_offset=sync_offset
    )
    ds_with_offset.set_num_cores(4)
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == -2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (4,) + tuple(ds_with_offset.shape.sig),
        sig_dims=ds_with_offset.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds_with_offset.shape,
    )

    t0 = next(p0.get_tiles(tiling_scheme))
    assert tuple(t0.tile_slice.origin) == (2, 0, 0)

    for p in ds_with_offset.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p.slice.origin == (48, 0, 0)
    assert p.slice.shape[0] == 16

    result = lt_ctx.run_udf(dataset=default_seq, udf=udf)
    result = result['intensity'].raw_data[:default_seq._meta.image_count - abs(sync_offset)]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


def test_missing_frames(lt_ctx):
    nav_shape = (16, 8)

    ds = SEQDataSet(path=SEQ_TESTDATA_PATH, nav_shape=nav_shape)
    ds.set_num_cores(4)
    ds = ds.initialize(lt_ctx.executor)
    ds.check_valid()

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p._start_frame == 96
    assert p._num_frames == 32
    assert p.slice.origin == (96, 0, 0)
    assert p.slice.shape[0] == 32
    assert t.tile_slice.origin == (60, 0, 0)
    assert t.tile_slice.shape[0] == 4


def test_missing_data_with_positive_sync_offset(lt_ctx):
    nav_shape = (16, 8)
    sync_offset = 8

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

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p._start_frame == 104
    assert p._num_frames == 32
    assert t.tile_slice.origin == (52, 0, 0)
    assert t.tile_slice.shape[0] == 4


def test_missing_data_with_negative_sync_offset(lt_ctx):
    nav_shape = (16, 8)
    sync_offset = -8

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

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p._start_frame == 88
    assert p._num_frames == 32
    assert t.tile_slice.origin == (68, 0, 0)
    assert t.tile_slice.shape[0] == 4


def test_too_many_frames(lt_ctx):
    nav_shape = (4, 8)

    ds = SEQDataSet(path=SEQ_TESTDATA_PATH, nav_shape=nav_shape)
    ds.set_num_cores(4)
    ds = ds.initialize(lt_ctx.executor)
    ds.check_valid()

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass


def test_positive_sync_offset_with_roi(default_seq, lt_ctx):
    udf = SumSigUDF()
    result = lt_ctx.run_udf(dataset=default_seq, udf=udf)
    result = result['intensity'].raw_data

    nav_shape = (8, 8)
    sync_offset = 2

    ds_with_offset = lt_ctx.load(
        "seq", path=SEQ_TESTDATA_PATH, nav_shape=nav_shape, sync_offset=sync_offset
    )

    roi = np.random.choice([False], (8, 8))
    roi[0:1] = True

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
    result_with_offset = result_with_offset['intensity'].raw_data

    assert np.allclose(result[sync_offset:8 + sync_offset], result_with_offset)


def test_negative_sync_offset_with_roi(default_seq, lt_ctx):
    udf = SumSigUDF()
    result = lt_ctx.run_udf(dataset=default_seq, udf=udf)
    result = result['intensity'].raw_data

    nav_shape = (8, 8)
    sync_offset = -2

    ds_with_offset = lt_ctx.load(
        "seq", path=SEQ_TESTDATA_PATH, nav_shape=nav_shape, sync_offset=sync_offset
    )

    roi = np.random.choice([False], (8, 8))
    roi[0:1] = True

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
    result_with_offset = result_with_offset['intensity'].raw_data

    assert np.allclose(result[:8 + sync_offset], result_with_offset[abs(sync_offset):])


def test_offset_smaller_than_image_count(lt_ctx):
    nav_shape = (8, 8)
    sync_offset = -65

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "seq",
            path=SEQ_TESTDATA_PATH,
            nav_shape=nav_shape,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-64, 64\), which is \(-image_count, image_count\)"
    )


def test_offset_greater_than_image_count(lt_ctx):
    nav_shape = (8, 8)
    sync_offset = 65

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "seq",
            path=SEQ_TESTDATA_PATH,
            nav_shape=nav_shape,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-64, 64\), which is \(-image_count, image_count\)"
    )


def test_reshape_nav(lt_ctx, default_seq):
    udf = SumSigUDF()

    ds_with_1d_nav = lt_ctx.load("seq", path=SEQ_TESTDATA_PATH, nav_shape=(64,))
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    result_with_2d_nav = lt_ctx.run_udf(dataset=default_seq, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load("seq", path=SEQ_TESTDATA_PATH, nav_shape=(2, 4, 8))
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


def test_reshape_different_shapes(lt_ctx, default_seq):
    udf = SumSigUDF()

    result = lt_ctx.run_udf(dataset=default_seq, udf=udf)
    result = result['intensity'].raw_data

    ds_1 = lt_ctx.load("seq", path=SEQ_TESTDATA_PATH, nav_shape=(3, 6))
    result_1 = lt_ctx.run_udf(dataset=ds_1, udf=udf)
    result_1 = result_1['intensity'].raw_data

    assert np.allclose(result_1, result[:3*6])


def test_incorrect_sig_shape(lt_ctx):
    nav_shape = (8, 8)
    sig_shape = (5, 5)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "seq",
            path=SEQ_TESTDATA_PATH,
            nav_shape=nav_shape,
            sig_shape=sig_shape
        )
    assert e.match(
        r"sig_shape must be of size: 16384"
    )


def test_scan_size_deprecation(lt_ctx):
    scan_size = (5, 5)

    with pytest.warns(FutureWarning):
        ds = lt_ctx.load(
            "seq",
            path=SEQ_TESTDATA_PATH,
            scan_size=scan_size,
        )
    assert tuple(ds.shape) == (5, 5, 128, 128)


def test_detect_unicode_error(raw_with_zeros, lt_ctx):
    path = raw_with_zeros._path
    SEQDataSet.detect_params(path, InlineJobExecutor())


# from utils import dataset_correction_verification
# FIXME test with actual test file
