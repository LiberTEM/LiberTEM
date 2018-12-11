import os

import numpy as np
import pytest

from libertem.io.dataset.k2is import K2ISDataSet
from libertem.job.masks import ApplyMasksJob
from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.raw import PickFrameAnalysis

K2IS_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                                  'data', 'Capture52', 'Capture52_.gtg')
HAVE_K2IS_TESTDATA = os.path.exists(K2IS_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_K2IS_TESTDATA, reason="need K2IS testdata")  # NOQA


@pytest.fixture
def default_k2is():
    scan_size = (34, 35)
    return K2ISDataSet(path=K2IS_TESTDATA_PATH, scan_size=scan_size)


def test_detect():
    params = K2ISDataSet.detect_params(K2IS_TESTDATA_PATH)
    assert params == {
        "path": K2IS_TESTDATA_PATH,
    }


def test_simple_open(default_k2is):
    assert tuple(default_k2is.effective_shape) == (34, 35, 1860, 2048)
    assert tuple(default_k2is.shape) == (34 * 35, 1860, 2048)


def test_check_valid(default_k2is):
    assert default_k2is.check_valid()


def test_sync(default_k2is):
    p = next(default_k2is.get_partitions())
    sector = p._get_sector()
    first_block = next(sector.get_blocks())
    assert first_block.header['frame_id'] == 60


def test_read(default_k2is):
    partitions = default_k2is.get_partitions()
    p = next(partitions)
    # NOTE: partition shape may change in the future
    assert tuple(p.shape) == (595, 2 * 930, 256)
    tiles = p.get_tiles()
    t = next(tiles)
    # we get 3D tiles here, because K2IS partitions are inherently 3D
    assert tuple(t.tile_slice.shape) == (16, 930, 16)


@pytest.mark.slow
def test_apply_mask_job(default_k2is, lt_ctx):
    mask = np.ones((1860, 2048))

    job = ApplyMasksJob(dataset=default_k2is, mask_factories=[lambda: mask])
    out = job.get_result_buffer()

    executor = InlineJobExecutor()

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(out)

    results = lt_ctx.run(job)
    assert results[0].shape == (34 * 35,)
    # there should be _something_ in each result pixel
    for px in results[0].reshape((-1,)):
        assert not np.isclose(px, 0)


@pytest.mark.slow
def test_apply_mask_analysis(default_k2is, lt_ctx):
    mask = np.ones((1860, 2048))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_k2is)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (34, 35)


@pytest.mark.slow
def test_sum_analysis(default_k2is, lt_ctx):
    analysis = lt_ctx.create_sum_analysis(dataset=default_k2is)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (1860, 2048)


def test_pick_job(default_k2is, lt_ctx):
    analysis = lt_ctx.create_pick_job(dataset=default_k2is, x=16, y=16)
    results = lt_ctx.run(analysis)
    assert results.shape == (1860, 2048)


def test_pick_analysis(default_k2is, lt_ctx):
    analysis = PickFrameAnalysis(dataset=default_k2is, parameters={"x": 16, "y": 16})
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (1860, 2048)
