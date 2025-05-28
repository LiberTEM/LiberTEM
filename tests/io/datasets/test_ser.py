import os

import pytest
import numpy as np

from libertem.udf.sum import SumUDF
from libertem.io.dataset.ser import SERDataSet
from libertem.udf.sumsigudf import SumSigUDF
from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape
from libertem.common.buffers import reshaped_view

from utils import dataset_correction_verification, get_testdata_path, ValidationUDF, roi_as_sparse

try:
    import hyperspy.api as hs
except ModuleNotFoundError:
    hs = None


SER_TESTDATA_PATH = os.path.join(get_testdata_path(), 'default.ser')
HAVE_SER_TESTDATA = os.path.exists(SER_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_SER_TESTDATA, reason="need SER testdata")


@pytest.fixture
def default_ser(lt_ctx):
    ds = SERDataSet(path=SER_TESTDATA_PATH, num_partitions=4)
    ds = ds.initialize(lt_ctx.executor)
    assert tuple(ds.shape) == (8, 35, 512, 512)
    return ds


@pytest.fixture(scope='module')
def default_ser_raw():
    res = hs.load(str(SER_TESTDATA_PATH))
    return res.data


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


@pytest.mark.skipif(hs is None, reason="No HyperSpy found")
def test_comparison(default_ser, default_ser_raw, lt_ctx_fast):
    udf = ValidationUDF(
        reference=reshaped_view(default_ser_raw, (-1, *tuple(default_ser.shape.sig)))
    )
    lt_ctx_fast.run_udf(udf=udf, dataset=default_ser)


@pytest.mark.skipif(hs is None, reason="No HyperSpy found")
def test_comparison_roi(default_ser, default_ser_raw, lt_ctx_fast):
    roi = np.random.choice(
        [True, False],
        size=tuple(default_ser.shape.nav),
        p=[0.5, 0.5]
    )
    udf = ValidationUDF(reference=default_ser_raw[roi])
    lt_ctx_fast.run_udf(udf=udf, dataset=default_ser, roi=roi)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_roi(lt_ctx, as_sparse):
    ds = lt_ctx.load("ser", path=SER_TESTDATA_PATH, num_partitions=2)
    roi = np.zeros(ds.shape.nav, dtype=bool)
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
    if as_sparse:
        roi = roi_as_sparse(roi)
    t1 = next(p1.get_tiles(tiling_scheme, roi=roi))
    assert t1.tile_slice.origin[0] == 1


def test_with_udf(lt_ctx):
    ds = lt_ctx.load("ser", path=SER_TESTDATA_PATH)
    udf = SumUDF()
    udf_results = lt_ctx.run_udf(udf=udf, dataset=ds)

    # compare result with a known-working variant:
    result = np.zeros(ds.shape.sig)
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


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
@pytest.mark.slow
def test_correction(lt_ctx, with_roi):
    ds = lt_ctx.load("ser", path=SER_TESTDATA_PATH)

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


def test_positive_sync_offset(default_ser, lt_ctx):
    udf = SumSigUDF()
    sync_offset = 2

    ds_with_offset = SERDataSet(
        path=SER_TESTDATA_PATH, sync_offset=sync_offset,
        num_partitions=4,
    )
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == 2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (1,) + tuple(ds_with_offset.shape.sig),
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

    assert p.slice.origin == (210, 0, 0)
    assert p.slice.shape[0] == 70

    result = lt_ctx.run_udf(dataset=default_ser, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
        :ds_with_offset._meta.image_count - sync_offset
    ]

    assert np.allclose(result, result_with_offset)


def test_negative_sync_offset(default_ser, lt_ctx):
    udf = SumSigUDF()
    sync_offset = -2

    ds_with_offset = SERDataSet(
        path=SER_TESTDATA_PATH, sync_offset=sync_offset,
        num_partitions=4,
    )
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == -2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (1,) + tuple(ds_with_offset.shape.sig),
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

    assert p.slice.origin == (210, 0, 0)
    assert p.slice.shape[0] == 70

    result = lt_ctx.run_udf(dataset=default_ser, udf=udf)
    result = result['intensity'].raw_data[:default_ser._meta.image_count - abs(sync_offset)]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


def test_positive_sync_offset_with_roi(default_ser, lt_ctx):
    udf = SumSigUDF()
    result = lt_ctx.run_udf(dataset=default_ser, udf=udf)
    result = result['intensity'].raw_data

    sync_offset = 2

    ds_with_offset = lt_ctx.load("ser", path=SER_TESTDATA_PATH, sync_offset=sync_offset)

    roi = np.random.choice([False], (8, 35))
    roi[0:1] = True

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
    result_with_offset = result_with_offset['intensity'].raw_data

    assert np.allclose(result[sync_offset:35 + sync_offset], result_with_offset)


def test_negative_sync_offset_with_roi(default_ser, lt_ctx):
    udf = SumSigUDF()
    result = lt_ctx.run_udf(dataset=default_ser, udf=udf)
    result = result['intensity'].raw_data

    sync_offset = -2

    ds_with_offset = lt_ctx.load(
        "ser", path=SER_TESTDATA_PATH, sync_offset=sync_offset
    )

    roi = np.random.choice([False], (8, 35))
    roi[0:1] = True

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
    result_with_offset = result_with_offset['intensity'].raw_data

    assert np.allclose(result[:35 + sync_offset], result_with_offset[abs(sync_offset):])


def test_missing_frames(lt_ctx):
    nav_shape = (10, 35)

    ds = SERDataSet(
        path=SER_TESTDATA_PATH,
        nav_shape=nav_shape,
        num_partitions=4,
    )
    ds = ds.initialize(lt_ctx.executor)

    tileshape = Shape(
        (1,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p._start_frame == 262
    assert p._num_frames == 88
    assert p.slice.origin == (262, 0, 0)
    assert p.slice.shape[0] == 88
    assert t.tile_slice.origin == (279, 0, 0)
    assert t.tile_slice.shape[0] == 1


def test_too_many_frames(lt_ctx):
    nav_shape = (7, 35)

    ds = SERDataSet(
        path=SER_TESTDATA_PATH,
        nav_shape=nav_shape,
        num_partitions=4,
    )
    ds = ds.initialize(lt_ctx.executor)

    tileshape = Shape(
        (1,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass


def test_offset_smaller_than_image_count(lt_ctx):
    sync_offset = -286

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "ser",
            path=SER_TESTDATA_PATH,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-280, 280\), which is \(-image_count, image_count\)"
    )


def test_offset_greater_than_image_count(lt_ctx):
    sync_offset = 286

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "ser",
            path=SER_TESTDATA_PATH,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-280, 280\), which is \(-image_count, image_count\)"
    )


def test_reshape_nav(default_ser, lt_ctx):
    udf = SumSigUDF()

    ds_with_1d_nav = lt_ctx.load("ser", path=SER_TESTDATA_PATH, nav_shape=(8,))
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    ds_with_2d_nav = lt_ctx.load("ser", path=SER_TESTDATA_PATH, nav_shape=(4, 2))
    result_with_2d_nav = lt_ctx.run_udf(dataset=ds_with_2d_nav, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load("ser", path=SER_TESTDATA_PATH, nav_shape=(2, 2, 2))
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


def test_incorrect_sig_shape(lt_ctx):
    sig_shape = (5, 5)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "ser",
            path=SER_TESTDATA_PATH,
            sig_shape=sig_shape
        )
    assert e.match(
        r"sig_shape must be of size: 262144"
    )


def test_scheme_too_large(default_ser):
    partitions = default_ser.get_partitions()
    p = next(partitions)
    depth = p.shape[0]

    # we make a tileshape that is too large for the partition here:
    tileshape = Shape(
        (depth + 1,) + tuple(default_ser.shape.sig),
        sig_dims=default_ser.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_ser.shape,
    )

    # tile shape is clamped to partition shape
    # (in SER adjusted to depth=1 at the moment):
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert tuple(t.tile_slice.shape)[0] <= depth


def test_bad_params(ds_params_tester, standard_bad_ds_params):
    args = ("ser", SER_TESTDATA_PATH)
    for params in standard_bad_ds_params:
        ds_params_tester(*args, **params)


def test_no_num_partitions(lt_ctx):
    ds = lt_ctx.load(
        "ser",
        path=SER_TESTDATA_PATH,
    )
    lt_ctx.run_udf(dataset=ds, udf=SumSigUDF())
