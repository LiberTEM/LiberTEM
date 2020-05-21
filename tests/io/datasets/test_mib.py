import os
import pickle
import json
from unittest import mock

import numpy as np
import pytest

from libertem.io.dataset.mib import MIBDataSet
from libertem.job.masks import ApplyMasksJob
from libertem.job.raw import PickFrameJob
from libertem.udf.raw import PickUDF
from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.raw import PickFrameAnalysis
from libertem.common import Slice, Shape
from libertem.udf.sumsigudf import SumSigUDF
from libertem.io.dataset.base import TilingScheme

from utils import dataset_correction_verification, get_testdata_path

MIB_TESTDATA_PATH = os.path.join(get_testdata_path(), 'default.mib')
HAVE_MIB_TESTDATA = os.path.exists(MIB_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_MIB_TESTDATA, reason="need .mib testdata")  # NOQA


@pytest.fixture
def default_mib(lt_ctx):
    nav_shape = (32, 32)
    ds = MIBDataSet(path=MIB_TESTDATA_PATH, nav_shape=nav_shape)
    ds.set_num_cores(4)
    ds = ds.initialize(lt_ctx.executor)
    return ds


def test_detect(lt_ctx):
    params = MIBDataSet.detect_params(MIB_TESTDATA_PATH, lt_ctx.executor)["parameters"]
    assert params == {
        "path": MIB_TESTDATA_PATH,
        "nav_shape": (1024,),
        "sig_shape": (256, 256)
    }


def test_simple_open(default_mib):
    assert tuple(default_mib.shape) == (32, 32, 256, 256)


def test_positive_sync_offset(default_mib, lt_ctx):
    udf = SumSigUDF()
    sync_offset = 2

    ds_with_offset = MIBDataSet(
        path=MIB_TESTDATA_PATH, nav_shape=(32, 32), sync_offset=sync_offset
    )
    ds_with_offset.set_num_cores(4)
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == 2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (16,) + tuple(ds_with_offset.shape.sig),
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

    assert p.slice.origin == (768, 0, 0)
    assert p.slice.shape[0] == 256

    result = lt_ctx.run_udf(dataset=default_mib, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
        :ds_with_offset._meta.image_count - sync_offset
    ]

    assert np.allclose(result, result_with_offset)


def test_negative_sync_offset(default_mib, lt_ctx):
    udf = SumSigUDF()
    sync_offset = -2

    ds_with_offset = MIBDataSet(
        path=MIB_TESTDATA_PATH, nav_shape=(32, 32), sync_offset=sync_offset
    )
    ds_with_offset.set_num_cores(4)
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == -2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (16,) + tuple(ds_with_offset.shape.sig),
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

    assert p.slice.origin == (768, 0, 0)
    assert p.slice.shape[0] == 256

    result = lt_ctx.run_udf(dataset=default_mib, udf=udf)
    result = result['intensity'].raw_data[:default_mib._meta.image_count - abs(sync_offset)]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


def test_offset_smaller_than_nav_shape(lt_ctx):
    nav_shape = (32, 32)
    sync_offset = -1030

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "mib",
            path=MIB_TESTDATA_PATH,
            nav_shape=nav_shape,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-1024, 1024\), which is \(-image_count, image_count\)"
        )


def test_offset_greater_than_nav_shape(lt_ctx):
    nav_shape = (32, 32)
    sync_offset = 1030

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "mib",
            path=MIB_TESTDATA_PATH,
            nav_shape=nav_shape,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-1024, 1024\), which is \(-image_count, image_count\)"
        )


@pytest.mark.with_numba
def test_read(default_mib):
    partitions = default_mib.get_partitions()
    p = next(partitions)
    assert len(p.shape) == 3
    assert tuple(p.shape[1:]) == (256, 256)

    tileshape = Shape(
        (3,) + tuple(default_mib.shape.sig),
        sig_dims=default_mib.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_mib.shape,
    )

    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    # we get 3D tiles here, because MIB partitions are inherently 3D
    assert tuple(t.tile_slice.shape) == (3, 256, 256)


def test_pickle_is_small(default_mib):
    pickled = pickle.dumps(default_mib)
    pickle.loads(pickled)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 2 * 1024


def test_apply_mask_on_mib_job(default_mib, lt_ctx):
    mask = np.ones((256, 256))

    job = ApplyMasksJob(dataset=default_mib, mask_factories=[lambda: mask])
    out = job.get_result_buffer()

    executor = InlineJobExecutor()

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.reduce_into_result(out)

    results = lt_ctx.run(job)
    assert results[0].shape == (1024,)


@pytest.mark.parametrize(
    'TYPE', ['JOB', 'UDF']
)
def test_apply_mask_analysis(default_mib, lt_ctx, TYPE):
    mask = np.ones((256, 256))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_mib)
    analysis.TYPE = TYPE
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (32, 32)


def test_sum_analysis(default_mib, lt_ctx):
    analysis = lt_ctx.create_sum_analysis(dataset=default_mib)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (256, 256)


def test_pick_job(default_mib, lt_ctx):
    analysis = lt_ctx.create_pick_job(dataset=default_mib, origin=(16, 16))
    results = lt_ctx.run(analysis)
    assert results.shape == (256, 256)


@pytest.mark.parametrize(
    'TYPE', ['JOB', 'UDF']
)
def test_pick_analysis(default_mib, lt_ctx, TYPE):
    analysis = PickFrameAnalysis(dataset=default_mib, parameters={"x": 16, "y": 16})
    analysis.TYPE = TYPE
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (256, 256)


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_correction(default_mib, lt_ctx, with_roi):
    ds = default_mib

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None
    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx, exclude=[(45, 144), (124, 30)])
    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


@pytest.mark.with_numba
def test_with_roi(default_mib, lt_ctx):
    udf = PickUDF()
    roi = np.zeros(default_mib.shape.nav, dtype=bool)
    roi[0] = 1
    res = lt_ctx.run_udf(udf=udf, dataset=default_mib, roi=roi)
    np.array(res['intensity']).shape == (1, 256, 256)


def test_crop_to(default_mib, lt_ctx):
    slice_ = Slice(shape=Shape((1024, 64, 64), sig_dims=2), origin=(0, 64, 64))
    job = PickFrameJob(dataset=default_mib, slice_=slice_)
    res = lt_ctx.run(job)
    assert res.shape == (1024, 64, 64)
    # TODO: check contents


def test_read_at_boundaries(default_mib, lt_ctx):
    nav_shape = (32, 32)
    ds_odd = MIBDataSet(path=MIB_TESTDATA_PATH, nav_shape=nav_shape)
    ds_odd = ds_odd.initialize(lt_ctx.executor)

    sumjob_odd = lt_ctx.create_sum_analysis(dataset=ds_odd)
    res_odd = lt_ctx.run(sumjob_odd)

    sumjob = lt_ctx.create_sum_analysis(dataset=default_mib)
    res = lt_ctx.run(sumjob)

    assert np.allclose(res[0].raw_data, res_odd[0].raw_data)


def test_diagnostics(default_mib):
    print(default_mib.diagnostics)


def test_cache_key_json_serializable(default_mib):
    json.dumps(default_mib.get_cache_key())


@pytest.mark.dist
def test_mib_dist(dist_ctx):
    nav_shape = (32, 32)
    ds = MIBDataSet(path=MIB_TESTDATA_PATH, nav_shape=nav_shape)
    ds = ds.initialize(dist_ctx.executor)
    analysis = dist_ctx.create_sum_analysis(dataset=ds)
    results = dist_ctx.run(analysis)
    assert results[0].raw_data.shape == (256, 256)


def test_too_many_files(lt_ctx):
    ds = MIBDataSet(path=MIB_TESTDATA_PATH, nav_shape=(32, 32))

    with mock.patch('libertem.io.dataset.mib.glob', side_effect=lambda p: [
            "/a/%d.mib" % i
            for i in range(256*256)
    ]):
        with pytest.warns(RuntimeWarning) as record:
            ds._filenames()

    assert len(record) == 1
    assert "Saving data in many small files" in record[0].message.args[0]


def test_not_too_many_files(lt_ctx):
    ds = MIBDataSet(path=MIB_TESTDATA_PATH, nav_shape=(32, 32))

    with mock.patch('libertem.io.dataset.mib.glob', side_effect=lambda p: [
            "/a/%d.mib" % i
            for i in range(256)
    ]):
        with pytest.warns(None) as record:
            ds._filenames()

    assert len(record) == 0


def test_reshape_nav(lt_ctx):
    udf = SumSigUDF()

    ds_with_1d_nav = lt_ctx.load("mib", path=MIB_TESTDATA_PATH, nav_shape=(8,))
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    ds_with_2d_nav = lt_ctx.load("mib", path=MIB_TESTDATA_PATH, nav_shape=(4, 2))
    result_with_2d_nav = lt_ctx.run_udf(dataset=ds_with_2d_nav, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load("mib", path=MIB_TESTDATA_PATH, nav_shape=(2, 2, 2))
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


def test_incorrect_sig_shape(lt_ctx):
    nav_shape = (32, 32)
    sig_shape = (5, 5)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "mib",
            path=MIB_TESTDATA_PATH,
            nav_shape=nav_shape,
            sig_shape=sig_shape
        )
    assert e.match(
        r"sig_shape must be of size: 65536"
    )


def test_scan_size_deprecation(lt_ctx):
    scan_size = (2, 2)

    with pytest.warns(FutureWarning):
        ds = lt_ctx.load(
            "mib",
            path=MIB_TESTDATA_PATH,
            scan_size=scan_size,
        )
    assert tuple(ds.shape) == (2, 2, 256, 256)
