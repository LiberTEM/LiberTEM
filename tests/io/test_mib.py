import os

import numpy as np
import pytest

from libertem.io.dataset.mib import MIBDataSet
from libertem.job.masks import ApplyMasksJob
from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.raw import PickFrameAnalysis

MIB_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'default.mib')
HAVE_MIB_TESTDATA = os.path.exists(MIB_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_MIB_TESTDATA, reason="need .mib testdata")  # NOQA


@pytest.fixture
def default_mib():
    scan_size = (32, 32)
    return MIBDataSet(path=MIB_TESTDATA_PATH, tileshape=(1, 8, 256, 256), scan_size=scan_size)


def test_detect():
    params = MIBDataSet.detect_params(MIB_TESTDATA_PATH)
    assert params == {
        "path": MIB_TESTDATA_PATH,
        "tileshape": (1, 8, 256, 256)
    }


def test_simple_open(default_mib):
    assert tuple(default_mib.shape) == (32, 32, 256, 256)
    assert tuple(default_mib.raw_shape) == (32 * 32, 256, 256)


def test_check_valid(default_mib):
    default_mib.check_valid()


def test_read(default_mib):
    partitions = default_mib.get_partitions()
    p = next(partitions)
    assert tuple(p.shape) == (32 * 32, 256, 256)
    tiles = p.get_tiles()
    t = next(tiles)
    # we get 3D tiles here, because MIB partitions are inherently 3D
    assert tuple(t.tile_slice.shape) == (8, 256, 256)


def test_pickle_doesnt_pickle_headers(default_mib):
    """
    warning, testing an implementation detail here, but useful to see if it actually works
    """
    import pickle
    pickled = pickle.dumps(default_mib)
    assert len(default_mib._headers) > 0
    reloaded = pickle.loads(pickled)
    assert len(reloaded._headers) == 0


def test_apply_mask_on_mib_job(default_mib, lt_ctx):
    mask = np.ones((256, 256))

    job = ApplyMasksJob(dataset=default_mib, mask_factories=[lambda: mask])
    out = job.get_result_buffer()

    executor = InlineJobExecutor()

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(out)

    results = lt_ctx.run(job)
    assert results[0].shape == (1024,)


def test_apply_mask_analysis(default_mib, lt_ctx):
    mask = np.ones((256, 256))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_mib)
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


def test_pick_analysis(default_mib, lt_ctx):
    analysis = PickFrameAnalysis(dataset=default_mib, parameters={"x": 16, "y": 16})
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (256, 256)
