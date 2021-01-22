import os

import numpy as np
import pytest
import mrcfile

from libertem.io.dataset.mrc import MRCDataSet
from libertem.udf.sumsigudf import SumSigUDF
from libertem.io.dataset.base import BufferedBackend

from utils import dataset_correction_verification, get_testdata_path


MRC_TESTDATA_PATH = os.path.join(
    get_testdata_path(), 'mrc', '20200821_92978_movie.mrc',
)
HAVE_MRC_TESTDATA = os.path.exists(MRC_TESTDATA_PATH)

print(MRC_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_MRC_TESTDATA, reason="need .mrc testdata")  # NOQA


@pytest.fixture
def default_mrc(lt_ctx):
    ds = lt_ctx.load("mrc", path=MRC_TESTDATA_PATH)
    return ds


@pytest.fixture
def buffered_mrc(lt_ctx):
    # FIXME: not yet supported
    ds = lt_ctx.load("mrc", path=MRC_TESTDATA_PATH, io_backend=BufferedBackend())
    return ds


@pytest.fixture(scope='module')
def default_mrc_raw():
    mrc = mrcfile.open(MRC_TESTDATA_PATH)
    return mrc.data


def test_simple_open(default_mrc):
    assert tuple(default_mrc.shape) == (4, 1024, 1024)


def test_check_valid(default_mrc):
    default_mrc.check_valid()


def test_read_roi(default_mrc, default_mrc_raw, lt_ctx):
    roi = np.zeros((4,), dtype=bool)
    roi[2] = 1
    sumj = lt_ctx.create_sum_analysis(dataset=default_mrc)
    sumres = lt_ctx.run(sumj, roi=roi)

    ref = default_mrc_raw[roi].sum(axis=0)

    assert np.allclose(sumres.intensity.raw_data, ref)


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_correction(default_mrc, lt_ctx, with_roi):
    ds = default_mrc

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


def test_detect_1(lt_ctx):
    fpath = MRC_TESTDATA_PATH
    assert MRCDataSet.detect_params(
        path=fpath,
        executor=lt_ctx.executor,
    )["parameters"] == {
        'path': fpath,
        'nav_shape': (4,),
        'sig_shape': (1024, 1024)
    }


def test_detect_2(lt_ctx):
    assert MRCDataSet.detect_params(
        path="nofile.someext",
        executor=lt_ctx.executor,
    ) is False


@pytest.mark.dist
def test_mrc_dist(dist_ctx):
    ds = MRCDataSet(path=MRC_TESTDATA_PATH)
    ds = ds.initialize(dist_ctx.executor)
    analysis = dist_ctx.create_sum_analysis(dataset=ds)
    roi = np.random.choice([True, False], size=4)
    results = dist_ctx.run(analysis, roi=roi)
    assert results[0].raw_data.shape == (1024, 1024)


def test_positive_sync_offset(lt_ctx):
    udf = SumSigUDF()
    sync_offset = 2

    ds = lt_ctx.load(
        "mrc", path=MRC_TESTDATA_PATH, nav_shape=(2, 2),
    )

    result = lt_ctx.run_udf(dataset=ds, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    ds_with_offset = lt_ctx.load(
        "mrc", path=MRC_TESTDATA_PATH, nav_shape=(2, 2), sync_offset=sync_offset
    )

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
        :ds_with_offset._meta.shape.nav.size - sync_offset
    ]

    assert np.allclose(result, result_with_offset)


def test_negative_sync_offset(default_mrc, lt_ctx):
    udf = SumSigUDF()
    sync_offset = -2

    ds = lt_ctx.load(
        "mrc", path=MRC_TESTDATA_PATH, nav_shape=(2, 2),
    )

    result = lt_ctx.run_udf(dataset=ds, udf=udf)
    result = result['intensity'].raw_data[:ds._meta.shape.nav.size - abs(sync_offset)]

    ds_with_offset = lt_ctx.load(
        "mrc", path=MRC_TESTDATA_PATH, nav_shape=(2, 2), sync_offset=sync_offset
    )

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


def test_offset_smaller_than_image_count(lt_ctx):
    sync_offset = -20

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "mrc",
            path=MRC_TESTDATA_PATH,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-4, 4\), which is \(-image_count, image_count\)"
    )


def test_offset_greater_than_image_count(lt_ctx):
    sync_offset = 20

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "mrc",
            path=MRC_TESTDATA_PATH,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-4, 4\), which is \(-image_count, image_count\)"
    )


def test_reshape_nav(lt_ctx):
    udf = SumSigUDF()

    ds_with_1d_nav = lt_ctx.load("mrc", path=MRC_TESTDATA_PATH, nav_shape=(4,))
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    ds_with_2d_nav = lt_ctx.load("mrc",  path=MRC_TESTDATA_PATH, nav_shape=(2, 2,))
    result_with_2d_nav = lt_ctx.run_udf(dataset=ds_with_2d_nav, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load("mrc",  path=MRC_TESTDATA_PATH, nav_shape=(1, 2, 2))
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


def test_incorrect_sig_shape(lt_ctx):
    sig_shape = (5, 5)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "mrc",
            path=MRC_TESTDATA_PATH,
            sig_shape=sig_shape
        )
    assert e.match(
        r"sig_shape must be of size: 1048576"
    )
