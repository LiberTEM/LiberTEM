import pickle

import numpy as np

from libertem.job.masks import ApplyMasksJob
from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.raw import PickFrameAnalysis


def test_simple_open(default_raw):
    assert tuple(default_raw.shape) == (16, 16, 128, 128)


def test_check_valid(default_raw):
    default_raw.check_valid()


def test_read(default_raw):
    partitions = default_raw.get_partitions()
    p = next(partitions)
    # FIXME: partition shape can vary by number of cores
    # assert tuple(p.shape) == (2, 16, 128, 128)
    tiles = p.get_tiles()
    t = next(tiles)
    # default tileshape -> whole partition
    assert tuple(t.tile_slice.shape) == tuple(p.shape)


def test_pickle_is_small(default_raw):
    pickled = pickle.dumps(default_raw)
    pickle.loads(pickled)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 2 * 1024


def test_apply_mask_on_raw_job(default_raw, lt_ctx):
    mask = np.ones((128, 128))

    job = ApplyMasksJob(dataset=default_raw, mask_factories=[lambda: mask])
    out = job.get_result_buffer()

    executor = InlineJobExecutor()

    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.reduce_into_result(out)

    results = lt_ctx.run(job)
    # FIXME: should the result here be 1D or 2D?
    # currently, for inherently 4D datasets it is 2D, and for 3D datasets
    # it is 1D. make this consistent?
    assert results[0].shape == (16, 16)


def test_apply_mask_analysis(default_raw, lt_ctx):
    mask = np.ones((128, 128))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_raw)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (16, 16)


def test_sum_analysis(default_raw, lt_ctx):
    analysis = lt_ctx.create_sum_analysis(dataset=default_raw)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)


def test_pick_job(default_raw, lt_ctx):
    analysis = lt_ctx.create_pick_job(dataset=default_raw, origin=(16, 16))
    results = lt_ctx.run(analysis)
    assert results.shape == (128, 128)


def test_pick_analysis(default_raw, lt_ctx):
    analysis = PickFrameAnalysis(dataset=default_raw, parameters={"x": 16, "y": 16})
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)
