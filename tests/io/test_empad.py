import os
import pickle

import pytest
import numpy as np

from libertem.job.masks import ApplyMasksJob
from libertem.job.raw import PickFrameJob
from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.raw import PickFrameAnalysis
from libertem.io.dataset.base import DataSetException
from libertem.io.dataset.empad import EMPADDataSet
from libertem.common import Slice, Shape
from utils import _mk_random

EMPAD_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'EMPAD')
EMPAD_RAW = os.path.join(EMPAD_TESTDATA_PATH, 'scan_11_x4_y4.raw')
EMPAD_XML = os.path.join(EMPAD_TESTDATA_PATH, 'acquisition_12_pretty.xml')
HAVE_EMPAD_TESTDATA = os.path.exists(EMPAD_RAW) and os.path.exists(EMPAD_XML)

pytestmark = pytest.mark.skipif(not HAVE_EMPAD_TESTDATA, reason="need EMPAD testdata")  # NOQA


@pytest.fixture
def default_empad():
    ds = EMPADDataSet(
        path=EMPAD_XML,
    )
    ds = ds.initialize()
    yield ds


@pytest.fixture(scope='session')
def random_empad(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/empad-test-default.raw'
    data = _mk_random(size=(16, 16, 130, 128), dtype='float32')
    data.tofile(str(filename))
    del data
    ds = EMPADDataSet(
        path=str(filename),
        scan_size=(16, 16),
    )
    ds = ds.initialize()
    yield ds


def test_simple_open(default_empad):
    assert tuple(default_empad.shape) == (4, 4, 128, 128)


def test_check_valid(default_empad):
    assert default_empad.check_valid()


def test_check_valid_random(random_empad):
    assert random_empad.check_valid()


def test_read_random(random_empad):
    partitions = random_empad.get_partitions()
    p = next(partitions)
    # FIXME: partition shape can vary by number of cores
    # assert tuple(p.shape) == (2, 16, 128, 128)
    tiles = p.get_tiles()
    t = next(tiles)

    # ~1MB
    assert tuple(t.tile_slice.shape) == (16, 128, 128)


def test_read(default_empad):
    partitions = default_empad.get_partitions()
    p = next(partitions)
    # FIXME: partition shape can vary by number of cores
    # assert tuple(p.shape) == (2, 16, 128, 128)
    tiles = p.get_tiles()
    t = next(tiles)

    # ~1MB
    assert tuple(t.tile_slice.shape) == (16, 128, 128)


def test_pickle_is_small(default_empad):
    pickled = pickle.dumps(default_empad)
    pickle.loads(pickled)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 2 * 1024


def test_apply_mask_on_empad_job(default_empad, lt_ctx):
    mask = np.ones((128, 128))

    job = ApplyMasksJob(dataset=default_empad, mask_factories=[lambda: mask])
    out = job.get_result_buffer()

    executor = InlineJobExecutor()

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.reduce_into_result(out)

    results = lt_ctx.run(job)
    assert results[0].shape == (4 * 4,)
    assert np.count_nonzero(results[0]) > 0


def test_apply_mask_analysis(default_empad, lt_ctx):
    mask = np.ones((128, 128))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_empad)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (4, 4)
    assert np.count_nonzero(results[0].raw_data) > 0


def test_sum_analysis(default_empad, lt_ctx):
    analysis = lt_ctx.create_sum_analysis(dataset=default_empad)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)
    assert np.count_nonzero(results[0].raw_data) > 0


def test_pick_job(default_empad, lt_ctx):
    analysis = lt_ctx.create_pick_job(dataset=default_empad, origin=(3,))
    results = lt_ctx.run(analysis)
    assert results.shape == (128, 128)
    assert np.count_nonzero(results[0]) > 0


def test_pick_analysis(default_empad, lt_ctx):
    analysis = PickFrameAnalysis(dataset=default_empad, parameters={"x": 2, "y": 2})
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)
    assert np.count_nonzero(results[0].raw_data) > 0


def test_invalid_size():
    ds = EMPADDataSet(
        path=EMPAD_RAW,
        scan_size=(4, 5),
    )
    ds = ds.initialize()
    with pytest.raises(DataSetException) as einfo:
        ds.check_valid()

    assert einfo.match("invalid filesize")


def test_nonexistent():
    ds = EMPADDataSet(
        path="/does/not/exist.raw",
        scan_size=(4, 4),
    )
    with pytest.raises(DataSetException) as einfo:
        ds = ds.initialize()
    assert einfo.match("No such file or directory")


def test_detect_fail():
    # does not exist:
    assert not EMPADDataSet.detect_params("/does/not/exist.raw")
    # exists but we can't detect any parameters (and we don't know if it even is an EMPAD file)
    assert not EMPADDataSet.detect_params(EMPAD_RAW)


def test_crop_to(default_empad, lt_ctx):
    slice_ = Slice(shape=Shape((4, 64, 64), sig_dims=2), origin=(0, 64, 64))
    job = PickFrameJob(dataset=default_empad, slice_=slice_)
    res = lt_ctx.run(job)
    assert res.shape == (4, 64, 64)
    assert np.count_nonzero(res) > 0
