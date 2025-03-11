import os
import sys
import json
import random

import pytest
import numpy as np
import cloudpickle

from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.raw import PickFrameAnalysis
from libertem.io.dataset.base import (
    DataSetException, TilingScheme, BufferedBackend, MMapBackend, DirectBackend
)
from libertem.io.dataset.empad import EMPADDataSet, get_params_from_xml
from libertem.common import Shape
from libertem.common.buffers import reshaped_view
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.raw import PickUDF
from utils import _mk_random

from utils import dataset_correction_verification, get_testdata_path, ValidationUDF, roi_as_sparse

EMPAD_TESTDATA_PATH = os.path.join(get_testdata_path(), 'EMPAD')
EMPAD_RAW = os.path.join(EMPAD_TESTDATA_PATH, 'scan_11_x4_y4.raw')
EMPAD_XML = os.path.join(EMPAD_TESTDATA_PATH, 'acquisition_12_pretty.xml')
EMPAD_XML_2 = os.path.join(EMPAD_TESTDATA_PATH, 'acquisition_12_pretty.xml')
EMPAD_XML_SERIES = os.path.join(EMPAD_TESTDATA_PATH, 'acquisition_series_pretty.xml')
EMPAD_XML_BROKEN_PARAMS = os.path.join(
    EMPAD_TESTDATA_PATH, 'Magical_DataSet/acquisition_1/acquisition_1.xml',
)
EMPAD_XML_BROKEN_PARAMS_BAD = os.path.join(
    EMPAD_TESTDATA_PATH, 'Magical_DataSet/acquisition_1/acquisition_1_bad.xml',
)
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


@pytest.fixture
def direct_empad(lt_ctx):
    direct = DirectBackend()
    return lt_ctx.load(
        "empad",
        path=EMPAD_XML,
        io_backend=direct,
    )


@pytest.fixture(scope='module')
def default_empad_raw():
    raw_data = np.memmap(
        EMPAD_RAW,
        shape=(4, 4, 130, 128),
        dtype=np.float32,
        mode='r'
    )
    return raw_data[:, :, :128, :]


def test_new_empad_xml(lt_ctx):
    ds = lt_ctx.load("empad", path=EMPAD_XML_2)
    assert ds.shape.to_tuple() == (4, 4, 128, 128)


def test_empad_broken_scan_parameters(lt_ctx):
    ds = lt_ctx.load("empad", path=EMPAD_XML_BROKEN_PARAMS)
    assert ds.shape.to_tuple() == (128, 16, 128, 128)


def test_empad_broken_scan_parameters_bad(lt_ctx):
    with pytest.raises(ValueError) as excinfo:
        lt_ctx.load("empad", path=EMPAD_XML_BROKEN_PARAMS_BAD)
    assert "(128, 17)" in str(excinfo.value)
    assert "(256, 256)" in str(excinfo.value)


def test_series_acquisition_xml():
    path_raw, nav_shape = get_params_from_xml(EMPAD_XML_SERIES)
    assert nav_shape == (1000,)


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


def test_comparison(default_empad, default_empad_raw, lt_ctx_fast):
    udf = ValidationUDF(
        reference=reshaped_view(default_empad_raw, (-1, *tuple(default_empad.shape.sig)))
    )
    lt_ctx_fast.run_udf(udf=udf, dataset=default_empad)


def test_comparison_roi(default_empad, default_empad_raw, lt_ctx_fast):
    roi = np.random.choice(
        [True, False],
        size=tuple(default_empad.shape.nav),
        p=[0.5, 0.5]
    )
    udf = ValidationUDF(reference=default_empad_raw[roi])
    lt_ctx_fast.run_udf(udf=udf, dataset=default_empad, roi=roi)


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

    for _ in tiles:
        pass


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

    for _ in tiles:
        pass


def test_scheme_too_large(default_empad):
    partitions = default_empad.get_partitions()
    p = next(partitions)
    depth = p.shape[0]

    # we make a tileshape that is too large for the partition here:
    tileshape = Shape(
        (depth + 1,) + tuple(default_empad.shape.sig),
        sig_dims=default_empad.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_empad.shape,
    )

    # tile shape is clamped to partition shape:
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == tuple((depth,) + default_empad.shape.sig)

    for _ in tiles:
        pass


def test_pickle_is_small(default_empad):
    pickled = cloudpickle.dumps(default_empad)
    cloudpickle.loads(pickled)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 2 * 1024


def test_apply_mask_analysis(default_empad, lt_ctx):
    mask = np.ones((128, 128))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_empad)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (4, 4)
    assert np.count_nonzero(results[0].raw_data) > 0


def test_sum_analysis(default_empad, lt_ctx):
    analysis = lt_ctx.create_sum_analysis(dataset=default_empad)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)
    assert np.count_nonzero(results[0].raw_data) > 0


def test_pick_analysis(default_empad, lt_ctx):
    analysis = PickFrameAnalysis(dataset=default_empad, parameters={"x": 2, "y": 2})
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


@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="No support for direct I/O on Mac OS X"
)
def test_compare_direct_to_mmap(lt_ctx, default_empad, direct_empad):
    y = random.choice(range(default_empad.shape.nav[0]))
    x = random.choice(range(default_empad.shape.nav[1]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_empad,
        x=x, y=y,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=direct_empad,
        x=x, y=y,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_compare_backends_sparse(lt_ctx, default_empad, buffered_empad, as_sparse):
    roi = np.zeros(default_empad.shape.nav, dtype=bool).reshape((-1,))
    roi[0] = True
    roi[1] = True
    roi[8] = True
    roi[-1] = True
    if as_sparse:
        roi = roi_as_sparse(roi)
    mm_f0 = lt_ctx.run_udf(dataset=default_empad, udf=PickUDF(), roi=roi)['intensity']
    buffered_f0 = lt_ctx.run_udf(dataset=buffered_empad, udf=PickUDF(), roi=roi)['intensity']

    assert np.allclose(mm_f0, buffered_f0)


def test_bad_params(ds_params_tester, standard_bad_ds_params):
    args = ("empad", EMPAD_XML)
    for params in standard_bad_ds_params:
        ds_params_tester(*args, **params)


def test_num_partitions(lt_ctx):
    ds = lt_ctx.load(
        "empad",
        path=EMPAD_XML,
        num_partitions=5,
    )
    assert len(list(ds.get_partitions())) == 5
