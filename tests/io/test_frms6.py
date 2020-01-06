import os
import pickle
import json

import pytest

from libertem.io.dataset.frms6 import FRMS6DataSet
from libertem.analysis.raw import PickFrameAnalysis
from libertem.analysis.sum import SumAnalysis

FRMS6_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                                 'data', 'frms6', 'C16_15_24_151203_019.hdr')
HAVE_FRMS6_TESTDATA = os.path.exists(FRMS6_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_FRMS6_TESTDATA, reason="need frms6 testdata")  # NOQA


@pytest.fixture
def default_frms6(lt_ctx):
    ds = FRMS6DataSet(path=FRMS6_TESTDATA_PATH)
    ds = ds.initialize(lt_ctx.executor)
    return ds


@pytest.fixture
def dist_frms6(dist_ctx):
    path = "/data/frms6/C16_15_24_151203_019.hdr"
    ds = FRMS6DataSet(path=path)
    ds = ds.initialize(dist_ctx.executor)
    assert ds._fileset is not None
    return ds


def test_simple_open(default_frms6):
    assert tuple(default_frms6.shape) == (256, 256, 264, 264)


def test_detetct(lt_ctx):
    assert FRMS6DataSet.detect_params(FRMS6_TESTDATA_PATH, lt_ctx.executor) is not False


def test_check_valid(default_frms6):
    default_frms6.check_valid()


def test_sum_analysis(default_frms6, lt_ctx):
    roi = {
        "shape": "disk",
        "cx": 5,
        "cy": 6,
        "r": 7,
    }
    analysis = SumAnalysis(dataset=default_frms6, parameters={
        "roi": roi,
    })
    # not checking result yet, just making sure it doesn't crash:
    lt_ctx.run(analysis)


def test_pick_job(default_frms6, lt_ctx):
    analysis = lt_ctx.create_pick_job(dataset=default_frms6, origin=(16, 16))
    results = lt_ctx.run(analysis)
    assert results.shape == (264, 264)


def test_pick_analysis(default_frms6, lt_ctx):
    analysis = PickFrameAnalysis(dataset=default_frms6, parameters={"x": 16, "y": 16})
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (264, 264)


def test_pickle_is_small(default_frms6):
    pickled = pickle.dumps(default_frms6)
    pickle.loads(pickled)

    # because of the dark frame stuff, the dataset is actually quite large:
    assert len(pickled) < 80 * 1024


def test_cache_key_json_serializable(default_frms6):
    json.dumps(default_frms6.get_cache_key())


@pytest.mark.dist
def test_dist_process(dist_frms6, dist_ctx):
    roi = {
        "shape": "disk",
        "cx": 5,
        "cy": 6,
        "r": 7,
    }
    analysis = SumAnalysis(dataset=dist_frms6, parameters={"roi": roi})
    dist_ctx.run(analysis)


# TODO: gain map tests
# TODO: test load request message
# TODO: test error conditions
