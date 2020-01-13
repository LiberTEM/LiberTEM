import os
import glob
import hashlib

import numpy as np
import pytest

from libertem.io.dataset.dm import DMDataSet


DM_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dm')
HAVE_DM_TESTDATA = os.path.exists(DM_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_DM_TESTDATA, reason="need .dm4 testdata")  # NOQA


@pytest.fixture
def default_dm(lt_ctx):
    files = glob.glob(os.path.join(DM_TESTDATA_PATH, '*.dm4'))
    ds = lt_ctx.load("dm", files=files)
    return ds


def test_simple_open(default_dm):
    assert tuple(default_dm.shape) == (10, 3838, 3710)


def test_check_valid(default_dm):
    default_dm.check_valid()


def test_read_roi(default_dm, lt_ctx):
    roi = np.zeros((10,), dtype=bool)
    roi[5] = 1
    sumj = lt_ctx.create_sum_analysis(dataset=default_dm)
    sumres = lt_ctx.run(sumj, roi=roi)
    sha1 = hashlib.sha1()
    sha1.update(sumres.intensity.raw_data)
    assert sha1.hexdigest() == "20a43a7ff069896fb2c46b0d7fa35f71d97d76f9"


def test_detect_1(lt_ctx):
    fpath = os.path.join(DM_TESTDATA_PATH, '2018-7-17 15_29_0000.dm4')
    assert DMDataSet.detect_params(
        path=fpath,
        executor=lt_ctx.executor,
    ) == {
        'files': [fpath],
    }


def test_detect_2(lt_ctx):
    assert DMDataSet.detect_params(
        path="nofile.someext",
        executor=lt_ctx.executor,
    ) is False


def test_same_offset(lt_ctx):
    files = glob.glob(os.path.join(DM_TESTDATA_PATH, '*.dm4'))
    ds = lt_ctx.load("dm", files=files, same_offset=True)
    ds.check_valid()


def test_repr(default_dm):
    assert repr(default_dm) == "<DMDataSet for a stack of 10 files>"


@pytest.mark.dist
def test_dm_dist(dist_ctx):
    files = dist_ctx.executor.run_function(lambda: list(sorted(glob.glob("/data/dm/*.dm4"))))
    print(files)
    ds = DMDataSet(files=files)
    ds = ds.initialize(dist_ctx.executor)
    analysis = dist_ctx.create_sum_analysis(dataset=ds)
    roi = np.random.choice([True, False], size=len(files))
    results = dist_ctx.run(analysis, roi=roi)
    assert results[0].raw_data.shape == (3838, 3710)
