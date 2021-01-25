import os
import json
import pickle
import random

import pytest
import numpy as np

from libertem.job.masks import ApplyMasksJob
from libertem.job.raw import PickFrameJob
from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.raw import PickFrameAnalysis
from libertem.io.dataset.base import DataSetException, TilingScheme, BufferedBackend, MMapBackend
from libertem.io.dataset.empad import EMPADDataSet
from libertem.common import Slice, Shape
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.raw import PickUDF
from utils import _mk_random

from utils import dataset_correction_verification, get_testdata_path

EMPAD_TESTDATA_PATH = os.path.join(get_testdata_path(), 'EMPAD')
EMPAD_RAW = os.path.join(EMPAD_TESTDATA_PATH, 'scan_11_x4_y4.raw')
EMPAD_XML = os.path.join(EMPAD_TESTDATA_PATH, 'acquisition_12_pretty.xml')
EMPAD_XML_2 = os.path.join(EMPAD_TESTDATA_PATH, 'acquisition_12_pretty.xml')
HAVE_EMPAD_TESTDATA = os.path.exists(EMPAD_RAW) and os.path.exists(EMPAD_XML)

pytestmark = pytest.mark.skipif(not HAVE_EMPAD_TESTDATA, reason="need EMPAD testdata")  # NOQA


@pytest.fixture
def default_empad(lt_ctx):
    ds = lt_ctx.load(
        "empad",
        path=EMPAD_XML,
        io_backend=MMapBackend(),
    )
    yield ds


@pytest.fixture
def buffered_empad(lt_ctx):
    buffered = BufferedBackend()
    return lt_ctx.load(
        "empad",
        path=EMPAD_XML,
        io_backend=buffered,
    )


def test_new_empad_xml():
    executor = InlineJobExecutor()
    ds = EMPADDataSet(
        path=EMPAD_XML_2,
    )
    ds = ds.initialize(executor)


@pytest.fixture(scope='session')
def random_empad(tmpdir_factory):
    executor = InlineJobExecutor()
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/empad-test-default.raw'
    data = _mk_random(size=(16, 16, 130, 128), dtype='float32')
    data.tofile(str(filename))
    del data
    ds = EMPADDataSet(
        path=str(filename),
        nav_shape=(16, 16),
    )
    ds = ds.initialize(executor)
    yield ds


def test_simple_open(default_empad):
    assert tuple(default_empad.shape) == (4, 4, 128, 128)


def test_check_valid(default_empad):
    assert default_empad.check_valid()


def test_check_valid_random(random_empad):
    assert random_empad.check_valid()


def test_read_random(random_empad):
    partitions = random_empad.get_partitions()
    p = next(partitions)
    # FIXME: partition shape can vary by number of cores
    # assert tuple(p.shape) == (2, 16, 128, 128)
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=Shape((16, 128, 128), sig_dims=2),
        dataset_shape=random_empad.shape,
    )
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)

    # ~1MB
    assert tuple(t.tile_slice.shape) == (16, 128, 128)


def test_read(default_empad):
    partitions = default_empad.get_partitions()
    p = next(partitions)
    # FIXME: partition shape can vary by number of cores
    # assert tuple(p.shape) == (2, 16, 128, 128)
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=Shape((16, 128, 128), sig_dims=2),
        dataset_shape=default_empad.shape,
    )
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)

    # ~1MB
    assert tuple(t.tile_slice.shape) == (16, 128, 128)


def test_pickle_is_small(default_empad):
    pickled = pickle.dumps(default_empad)
    pickle.loads(pickled)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 2 * 1024


def test_apply_mask_on_empad_job(default_empad, lt_ctx):
    mask = np.ones((128, 128))

    job = ApplyMasksJob(dataset=default_empad, mask_factories=[lambda: mask])
    out = job.get_result_buffer()

    executor = InlineJobExecutor()

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.reduce_into_result(out)

    results = lt_ctx.run(job)
    assert results[0].shape == (4 * 4,)
    assert np.count_nonzero(results[0]) > 0


@pytest.mark.parametrize(
    'TYPE', ['JOB', 'UDF']
)
def test_apply_mask_analysis(default_empad, lt_ctx, TYPE):
    mask = np.ones((128, 128))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_empad)
    analysis.TYPE = TYPE
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (4, 4)
    assert np.count_nonzero(results[0].raw_data) > 0


def test_sum_analysis(default_empad, lt_ctx):
    analysis = lt_ctx.create_sum_analysis(dataset=default_empad)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)
    assert np.count_nonzero(results[0].raw_data) > 0


def test_pick_job(default_empad, lt_ctx):
    analysis = lt_ctx.create_pick_job(dataset=default_empad, origin=(3,))
    results = lt_ctx.run(analysis)
    assert results.shape == (128, 128)
    assert np.count_nonzero(results[0]) > 0


@pytest.mark.parametrize(
    'TYPE', ['JOB', 'UDF']
)
def test_pick_analysis(default_empad, lt_ctx, TYPE):
    analysis = PickFrameAnalysis(dataset=default_empad, parameters={"x": 2, "y": 2})
    analysis.TYPE = TYPE
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)
    assert np.count_nonzero(results[0].raw_data) > 0


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_correction(default_empad, lt_ctx, with_roi):
    ds = default_empad

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


def test_nonexistent():
    ds = EMPADDataSet(
        path="/does/not/exist.raw",
        nav_shape=(4, 4),
    )
    with pytest.raises(DataSetException) as einfo:
        ds = ds.initialize(InlineJobExecutor())
    assert einfo.match("could not open file /does/not/exist.raw")


def test_detect_fail():
    executor = InlineJobExecutor()
    # does not exist:
    assert not EMPADDataSet.detect_params("/does/not/exist.raw", executor=executor)
    # exists but we can't detect any parameters (and we don't know if it even is an EMPAD file)
    assert not EMPADDataSet.detect_params(EMPAD_RAW, executor=executor)


def test_crop_to(default_empad, lt_ctx):
    slice_ = Slice(shape=Shape((4, 64, 64), sig_dims=2), origin=(0, 64, 64))
    job = PickFrameJob(dataset=default_empad, slice_=slice_)
    res = lt_ctx.run(job)
    assert res.shape == (4, 64, 64)
    assert np.count_nonzero(res) > 0


def test_cache_key_json_serializable(default_empad):
    json.dumps(default_empad.get_cache_key())


@pytest.mark.dist
def test_empad_dist(dist_ctx):
    ds = EMPADDataSet(path=EMPAD_XML)
    ds = ds.initialize(dist_ctx.executor)
    analysis = dist_ctx.create_sum_analysis(dataset=ds)
    results = dist_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)


def test_positive_sync_offset(default_empad, lt_ctx):
    udf = SumSigUDF()
    sync_offset = 2

    ds_with_offset = lt_ctx.load(
        "empad", path=EMPAD_XML, sync_offset=sync_offset
    )

    result = lt_ctx.run_udf(dataset=default_empad, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
        :ds_with_offset._meta.image_count - sync_offset
    ]

    assert np.allclose(result, result_with_offset)


def test_negative_sync_offset(default_empad, lt_ctx):
    udf = SumSigUDF()
    sync_offset = -2

    ds_with_offset = lt_ctx.load(
        "empad", path=EMPAD_XML, sync_offset=sync_offset
    )

    result = lt_ctx.run_udf(dataset=default_empad, udf=udf)
    result = result['intensity'].raw_data[:default_empad._meta.image_count - abs(sync_offset)]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


def test_positive_sync_offset_with_roi(default_empad, lt_ctx):
    udf = SumSigUDF()
    result = lt_ctx.run_udf(dataset=default_empad, udf=udf)
    result = result['intensity'].raw_data

    sync_offset = 2

    ds_with_offset = lt_ctx.load(
        "empad", path=EMPAD_XML, sync_offset=sync_offset
    )

    roi = np.random.choice([False], (4, 4))
    roi[0:1] = True

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
    result_with_offset = result_with_offset['intensity'].raw_data

    assert np.allclose(result[sync_offset:4 + sync_offset], result_with_offset)


def test_negative_sync_offset_with_roi(default_empad, lt_ctx):
    udf = SumSigUDF()
    result = lt_ctx.run_udf(dataset=default_empad, udf=udf)
    result = result['intensity'].raw_data

    sync_offset = -2

    ds_with_offset = lt_ctx.load(
        "empad", path=EMPAD_XML, sync_offset=sync_offset
    )

    roi = np.random.choice([False], (4, 4))
    roi[0:1] = True

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
    result_with_offset = result_with_offset['intensity'].raw_data

    assert np.allclose(result[:4 + sync_offset], result_with_offset[abs(sync_offset):])


def test_offset_smaller_than_image_count(lt_ctx):
    sync_offset = -20

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "empad",
            path=EMPAD_XML,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-16, 16\), which is \(-image_count, image_count\)"
    )


def test_offset_greater_than_image_count(lt_ctx):
    sync_offset = 20

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "empad",
            path=EMPAD_XML,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-16, 16\), which is \(-image_count, image_count\)"
    )


def test_reshape_nav(lt_ctx):
    udf = SumSigUDF()

    ds_with_1d_nav = lt_ctx.load("empad", path=EMPAD_XML, nav_shape=(8,))
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    ds_with_2d_nav = lt_ctx.load("empad", path=EMPAD_XML, nav_shape=(4, 2))
    result_with_2d_nav = lt_ctx.run_udf(dataset=ds_with_2d_nav, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load("empad", path=EMPAD_XML, nav_shape=(2, 2, 2))
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


@pytest.mark.parametrize(
    "sig_shape", ((16384,), (128, 8, 16))
)
def test_reshape_sig(lt_ctx, default_empad, sig_shape):
    udf = SumSigUDF()

    result = lt_ctx.run_udf(dataset=default_empad, udf=udf)
    result = result['intensity'].raw_data

    ds_1 = lt_ctx.load("empad", path=EMPAD_XML, sig_shape=sig_shape)
    result_1 = lt_ctx.run_udf(dataset=ds_1, udf=udf)
    result_1 = result_1['intensity'].raw_data

    assert np.allclose(result, result_1)


def test_incorrect_sig_shape(lt_ctx):
    sig_shape = (5, 5)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "empad",
            path=EMPAD_XML,
            sig_shape=sig_shape
        )
    assert e.match(
        r"sig_shape must be of size: 16384"
    )


def test_scan_size_deprecation(lt_ctx):
    scan_size = (2, 2)

    with pytest.warns(FutureWarning):
        ds = lt_ctx.load(
            "empad",
            path=EMPAD_XML,
            scan_size=scan_size,
        )
    assert tuple(ds.shape) == (2, 2, 128, 128)


def test_compare_backends(lt_ctx, default_empad, buffered_empad):
    y = random.choice(range(default_empad.shape.nav[0]))
    x = random.choice(range(default_empad.shape.nav[1]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_empad,
        x=x, y=y,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=buffered_empad,
        x=x, y=y,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


def test_compare_backends_sparse(lt_ctx, default_empad, buffered_empad):
    roi = np.zeros(default_empad.shape.nav, dtype=np.bool).reshape((-1,))
    roi[0] = True
    roi[1] = True
    roi[8] = True
    roi[-1] = True
    mm_f0 = lt_ctx.run_udf(dataset=default_empad, udf=PickUDF(), roi=roi)['intensity']
    buffered_f0 = lt_ctx.run_udf(dataset=buffered_empad, udf=PickUDF(), roi=roi)['intensity']

    assert np.allclose(mm_f0, buffered_f0)
