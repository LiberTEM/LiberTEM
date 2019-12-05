import os
import glob
import hashlib

import numpy as np
import pytest


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
