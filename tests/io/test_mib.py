import os

import numpy as np
import pytest

from libertem.io.dataset.mib import MIBDataSet
from libertem.job.masks import ApplyMasksJob
from libertem.executor.inline import InlineJobExecutor

MIB_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'default.mib')
HAVE_MIB_TESTDATA = os.path.exists(MIB_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_MIB_TESTDATA, reason="need .mib testdata")  # NOQA


@pytest.fixture
def default_mib():
    scan_size = (32, 32)
    return MIBDataSet(path=MIB_TESTDATA_PATH, tileshape=(1, 8, 256, 256), scan_size=scan_size)


@pytest.fixture
def another_mib():
    path = "/home/clausen/Data/Merlin/strain_karina/MOSFET/20181119 184223/default10.mib"
    scan_size = (256, 256)
    return MIBDataSet(path=path, tileshape=(1, 8, 256, 256), scan_size=scan_size)


def test_detect():
    params = MIBDataSet.detect_params(MIB_TESTDATA_PATH)
    assert params == {
        "path": MIB_TESTDATA_PATH,
        "tileshape": (1, 8, 256, 256)
    }


def test_simple_open(default_mib):
    assert tuple(default_mib.effective_shape) == (32, 32, 256, 256)
    assert tuple(default_mib.shape) == (32 * 32, 256, 256)


def test_check_valid(default_mib):
    default_mib.check_valid()


@pytest.mark.xfail(reason="_files_sorted cache is not working yet, needs a fix")
def test_files_sorted(default_mib):
    assert len(default_mib._files_sorted()) == 1
    assert default_mib._files_sorted.cache_info().misses == 1
    assert default_mib._files_sorted.cache_info().hits == 0
    # trigger cache reads (should be hits)
    assert len(default_mib._files_sorted()) == 1
    assert len(default_mib._files_sorted()) == 1
    assert default_mib._files_sorted.cache_info().misses == 1
    assert default_mib._files_sorted.cache_info().hits == 2


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


def test_apply_mask_on_mib_job_bleh(another_mib, lt_ctx):
    scan_size = (32, 32)
    tileshape = (1, 8, 256, 256)
    default_mib = MIBDataSet(path=MIB_TESTDATA_PATH, tileshape=tileshape, scan_size=scan_size)
    # first try on default_mib:
    mask = np.ones((256, 256))
    job = ApplyMasksJob(dataset=default_mib, mask_factories=[lambda: mask])
    out = job.get_result_buffer()

    executor = InlineJobExecutor()

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(out)

    results = lt_ctx.run(job)
    assert results[0].shape == (32 * 32,)

    # let's try on another file:
    mask = np.ones((256, 256))
    job = ApplyMasksJob(dataset=another_mib, mask_factories=[lambda: mask])
    out = job.get_result_buffer()

    executor = InlineJobExecutor()

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(out)

    results = lt_ctx.run(job)
    assert results[0].shape == (65536,)
