import os
import json
import pickle

import numpy as np
import pytest

from libertem.io.dataset.k2is import K2ISDataSet
from libertem.job.masks import ApplyMasksJob
from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.raw import PickFrameAnalysis
from libertem.common.buffers import BufferWrapper

K2IS_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                                  'data', 'Capture52', 'Capture52_.gtg')
HAVE_K2IS_TESTDATA = os.path.exists(K2IS_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_K2IS_TESTDATA, reason="need K2IS testdata")  # NOQA


@pytest.fixture
def default_k2is():
    ds = K2ISDataSet(path=K2IS_TESTDATA_PATH)
    ds.initialize()
    return ds


def test_detect():
    params = K2ISDataSet.detect_params(K2IS_TESTDATA_PATH)
    assert params == {
        "path": K2IS_TESTDATA_PATH,
    }


def test_simple_open(default_k2is):
    assert tuple(default_k2is.shape) == (34, 35, 1860, 2048)

    # shapes are JSON-encodable:
    json.dumps(tuple(default_k2is.shape))


def test_check_valid(default_k2is):
    assert default_k2is.check_valid()


def test_sync(default_k2is):
    p = next(default_k2is.get_partitions())
    with p._sectors[0] as sector:
        first_block = next(sector.get_blocks())
    assert first_block.header['frame_id'] == 60


def test_read(default_k2is):
    partitions = default_k2is.get_partitions()
    p = next(partitions)
    # NOTE: partition shape may change in the future
    assert tuple(p.shape) == (74, 2 * 930, 8 * 256)
    tiles = p.get_tiles()
    t = next(tiles)
    # we get 3D tiles here, because K2IS partitions are inherently 3D
    assert tuple(t.tile_slice.shape) == (16, 930, 16)


def test_read_full_frames(default_k2is):
    partitions = default_k2is.get_partitions()
    p = next(partitions)
    # NOTE: partition shape may change in the future
    assert tuple(p.shape) == (74, 2 * 930, 8 * 256)
    tiles = p.get_tiles(full_frames=True)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == (1, 1860, 2048)
    assert tuple(t.tile_slice.origin) == (0, 0, 0)

    for t in tiles:
        assert t.tile_slice.origin[0] < p.shape[0]


@pytest.mark.slow
def test_apply_mask_job(default_k2is, lt_ctx):
    mask = np.ones((1860, 2048))

    job = ApplyMasksJob(dataset=default_k2is, mask_factories=[lambda: mask])
    out = job.get_result_buffer()

    executor = InlineJobExecutor()

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.reduce_into_result(out)

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
    analysis = lt_ctx.create_pick_job(dataset=default_k2is, origin=(16, 16))
    results = lt_ctx.run(analysis)
    assert results.shape == (1860, 2048)


def test_pick_analysis(default_k2is, lt_ctx):
    analysis = PickFrameAnalysis(dataset=default_k2is, parameters={"x": 16, "y": 16})
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (1860, 2048)


def test_dataset_is_picklable(default_k2is):
    pickled = pickle.dumps(default_k2is)
    pickle.loads(pickled)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 2 * 1024


def test_partition_is_picklable(default_k2is):
    pickled = pickle.dumps(next(default_k2is.get_partitions()))
    pickle.loads(pickled)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 2 * 1024


def test_get_diags(default_k2is):
    diags = default_k2is.diagnostics

    # diags are JSON-encodable:
    json.dumps(diags)


@pytest.mark.slow
def test_udf_on_k2is(lt_ctx, default_k2is):
    def my_init(partition):
        return {}

    def my_buffers():
        return {
            'pixelsum': BufferWrapper(
                kind="nav", dtype="float32"
            )
        }

    def my_frame_fn(frame, pixelsum):
        pixelsum[:] = np.sum(frame)

    res = lt_ctx.run_udf(
        dataset=default_k2is,
        fn=my_frame_fn,
        init=my_init,
        make_buffers=my_buffers,
    )
    assert 'pixelsum' in res
    # print(data.shape, res['pixelsum'].data.shape)
    # assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(2, 3)))
