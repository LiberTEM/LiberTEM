import os
import pickle

import numpy as np
import pytest

from libertem.job.masks import ApplyMasksJob
from libertem.analysis.raw import PickFrameAnalysis
from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.blo import BloDataSet

BLO_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'default.blo')
HAVE_BLO_TESTDATA = os.path.exists(BLO_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_BLO_TESTDATA, reason="need .blo testdata")  # NOQA


@pytest.fixture()
def default_blo():
    ds = BloDataSet(
        path=str(BLO_TESTDATA_PATH),
        tileshape=(1, 8, 144, 144),
    )
    ds.initialize()
    return ds


def test_simple_open(default_blo):
    assert tuple(default_blo.shape) == (90, 121, 144, 144)


def test_check_valid(default_blo):
    assert default_blo.check_valid()


def test_detect():
    assert BloDataSet.detect_params(path=str(BLO_TESTDATA_PATH))


def test_read(default_blo):
    partitions = default_blo.get_partitions()
    p = next(partitions)
    # FIXME: partition shape can vary by number of cores
    # assert tuple(p.shape) == (11, 121, 144, 144)
    tiles = p.get_tiles()
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == (8, 144, 144)


def test_pickle_meta_is_small(default_blo):
    pickled = pickle.dumps(default_blo._meta)
    pickle.loads(pickled)
    assert len(pickled) < 512


def test_pickle_blofile_is_small(default_blo):
    pickled = pickle.dumps(default_blo._get_blo_file())
    pickle.loads(pickled)
    assert len(pickled) < 1024


def test_apply_mask_on_raw_job(default_blo, lt_ctx):
    mask = np.ones((144, 144))

    job = ApplyMasksJob(dataset=default_blo, mask_factories=[lambda: mask])
    out = job.get_result_buffer()

    executor = InlineJobExecutor()

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.reduce_into_result(out)

    results = lt_ctx.run(job)
    assert results[0].shape == (90 * 121,)


def test_apply_mask_analysis(default_blo, lt_ctx):
    mask = np.ones((144, 144))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_blo)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (90, 121)


def test_sum_analysis(default_blo, lt_ctx):
    analysis = lt_ctx.create_sum_analysis(dataset=default_blo)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (144, 144)


def test_pick_job(default_blo, lt_ctx):
    analysis = lt_ctx.create_pick_job(dataset=default_blo, origin=(16,))
    results = lt_ctx.run(analysis)
    assert results.shape == (144, 144)


def test_pick_analysis(default_blo, lt_ctx):
    analysis = PickFrameAnalysis(dataset=default_blo, parameters={"x": 16, "y": 16})
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (144, 144)
