import os
import hashlib

import numpy as np
import pytest

from libertem.io.dataset.mrc import MRCDataSet

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


def test_simple_open(default_mrc):
    assert tuple(default_mrc.shape) == (4, 1024, 1024)


def test_check_valid(default_mrc):
    default_mrc.check_valid()


def test_read_roi(default_mrc, lt_ctx):
    roi = np.zeros((4,), dtype=bool)
    roi[2] = 1
    sumj = lt_ctx.create_sum_analysis(dataset=default_mrc)
    sumres = lt_ctx.run(sumj, roi=roi)
    sha1 = hashlib.sha1()
    sha1.update(sumres.intensity.raw_data)
    assert sha1.hexdigest() == "c189a28e6a875a1c928b79557a46a2f100441c30"


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
