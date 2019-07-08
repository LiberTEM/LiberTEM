import os
import pickle

import numpy as np
import pytest

from libertem.io.dataset.mib import MIBDataSet
from libertem.job.masks import ApplyMasksJob
from libertem.job.raw import PickFrameJob
from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.raw import PickFrameAnalysis
from libertem.common import Slice, Shape

MIB_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'default.mib')
HAVE_MIB_TESTDATA = os.path.exists(MIB_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_MIB_TESTDATA, reason="need .mib testdata")  # NOQA


@pytest.fixture
def default_mib():
    scan_size = (32, 32)
    ds = MIBDataSet(path=MIB_TESTDATA_PATH, tileshape=(1, 3, 256, 256), scan_size=scan_size)
    ds = ds.initialize()
    return ds


def test_detect():
    params = MIBDataSet.detect_params(MIB_TESTDATA_PATH)
    assert params == {
        "path": MIB_TESTDATA_PATH,
        "tileshape": (1, 3, 256, 256)
    }


def test_simple_open(default_mib):
    assert tuple(default_mib.shape) == (32, 32, 256, 256)


def test_check_valid(default_mib):
    default_mib.check_valid()


@pytest.mark.xfail
def test_missing_frames(lt_ctx):
    """
    there can be some frames missing at the end
    """
    # one full row of additional frames in the data set than in the file
    scan_size = (33, 32)
    ds = MIBDataSet(path=MIB_TESTDATA_PATH, tileshape=(1, 3, 256, 256), scan_size=scan_size)
    ds = ds.initialize()
    ds.check_valid()

    for p in ds.get_partitions():
        for t in p.get_tiles():
            pass


def test_too_many_frames():
    """
    mib files can contain more frames than the intended scanning dimensions
    """
    # one full row of additional frames in the file
    scan_size = (31, 32)
    ds = MIBDataSet(path=MIB_TESTDATA_PATH, tileshape=(1, 3, 256, 256), scan_size=scan_size)
    ds = ds.initialize()
    ds.check_valid()

    for p in ds.get_partitions():
        for t in p.get_tiles():
            pass


def test_read(default_mib):
    partitions = default_mib.get_partitions()
    p = next(partitions)
    assert len(p.shape) == 3
    assert tuple(p.shape[1:]) == (256, 256)
    tiles = p.get_tiles()
    t = next(tiles)
    # we get 3D tiles here, because MIB partitions are inherently 3D
    assert tuple(t.tile_slice.shape) == (3, 256, 256)


def test_pickle_is_small(default_mib):
    pickled = pickle.dumps(default_mib)
    pickle.loads(pickled)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 2 * 1024


def test_apply_mask_on_mib_job(default_mib, lt_ctx):
    mask = np.ones((256, 256))

    job = ApplyMasksJob(dataset=default_mib, mask_factories=[lambda: mask])
    out = job.get_result_buffer()

    executor = InlineJobExecutor()

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.reduce_into_result(out)

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


def test_crop_to(default_mib, lt_ctx):
    slice_ = Slice(shape=Shape((1024, 64, 64), sig_dims=2), origin=(0, 64, 64))
    job = PickFrameJob(dataset=default_mib, slice_=slice_)
    res = lt_ctx.run(job)
    assert res.shape == (1024, 64, 64)
    # TODO: check contents


def test_read_at_boundaries(default_mib, lt_ctx):
    scan_size = (32, 32)
    ds_odd = MIBDataSet(path=MIB_TESTDATA_PATH, tileshape=(1, 7, 256, 256), scan_size=scan_size)
    ds_odd = ds_odd.initialize()

    sumjob_odd = lt_ctx.create_sum_analysis(dataset=ds_odd)
    res_odd = lt_ctx.run(sumjob_odd)

    sumjob = lt_ctx.create_sum_analysis(dataset=default_mib)
    res = lt_ctx.run(sumjob)

    assert np.allclose(res[0].raw_data, res_odd[0].raw_data)


def test_invalid_crop_full_frames_combo(default_mib, lt_ctx):
    slice_ = Slice(shape=Shape((1024, 64, 64), sig_dims=2), origin=(0, 64, 64))
    p = next(default_mib.get_partitions())
    with pytest.raises(ValueError):
        next(p.get_tiles(crop_to=slice_, full_frames=True))


def test_diagnostics(default_mib):
    print(default_mib.diagnostics)
