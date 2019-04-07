import pytest
import numpy as np

from libertem.io.dataset.raw_direct import DirectRawFileDataSet
from libertem.job.masks import ApplyMasksJob
from libertem.executor.inline import InlineJobExecutor
from utils import _mk_random


@pytest.fixture(scope='session')
def direct_raw(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-default'
    data = _mk_random(size=(16, 16, 128, 128), dtype='float32')
    data.tofile(str(filename))
    del data
    ds = DirectRawFileDataSet(
        path=str(filename),
        scan_size=(16, 16),
        dtype="float32",
        detector_size=(128, 128),
        stackheight=4,
    )
    ds = ds.initialize()
    yield ds


def test_simple_open(direct_raw):
    assert tuple(direct_raw.shape) == (16, 16, 128, 128)


def test_check_valid(direct_raw):
    direct_raw.check_valid()


def test_read(direct_raw):
    partitions = direct_raw.get_partitions()
    p = next(partitions)
    # FIXME: partition shape can vary by number of cores
    # assert tuple(p.shape) == (2, 16, 128, 128)
    tiles = p.get_tiles()
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == (4, 128, 128)


def test_apply_mask_on_raw_job(direct_raw, lt_ctx):
    mask = np.ones((128, 128))

    job = ApplyMasksJob(dataset=direct_raw, mask_factories=[lambda: mask])
    out = job.get_result_buffer()

    executor = InlineJobExecutor()

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.reduce_into_result(out)

    results = lt_ctx.run(job)
    assert results[0].shape == (16 * 16,)
