import os
import json
import pickle

import numpy as np
import pytest

from libertem.job.masks import ApplyMasksJob
from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.raw import PickFrameAnalysis
from libertem.io.dataset.raw import RAWDatasetParams


def test_simple_open(default_raw):
    assert tuple(default_raw.shape) == (16, 16, 128, 128)


@pytest.mark.parametrize(
    'TYPE', ['JOB', 'UDF']
)
def test_large_pick(large_raw, lt_ctx, TYPE):
    y, x = large_raw.shape.nav
    dy, dx = large_raw.shape.sig
    analysis = lt_ctx.create_pick_analysis(large_raw, y=y-1, x=x-1)
    analysis.TYPE = TYPE
    result = lt_ctx.run(analysis)
    assert result.intensity.raw_data.shape == tuple(large_raw.shape.sig)


def test_check_valid(default_raw):
    default_raw.check_valid()


def test_read(default_raw):
    partitions = default_raw.get_partitions()
    p = next(partitions)
    # FIXME: partition shape can vary by number of cores
    # assert tuple(p.shape) == (2, 16, 128, 128)
    tiles = p.get_tiles()
    t = next(tiles)

    # ~1MB
    assert tuple(t.tile_slice.shape) == (16, 128, 128)


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
    assert results[0].shape == (16 * 16,)


@pytest.mark.parametrize(
    'TYPE', ['JOB', 'UDF']
)
def test_apply_mask_analysis(default_raw, lt_ctx, TYPE):
    mask = np.ones((128, 128))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_raw)
    analysis.TYPE = TYPE
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (16, 16)


def test_sum_analysis(default_raw, lt_ctx):
    analysis = lt_ctx.create_sum_analysis(dataset=default_raw)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)


def test_pick_job(default_raw, lt_ctx):
    analysis = lt_ctx.create_pick_job(dataset=default_raw, origin=(16,))
    results = lt_ctx.run(analysis)
    assert results.shape == (128, 128)


@pytest.mark.parametrize(
    'TYPE', ['JOB', 'UDF']
)
def test_pick_analysis(default_raw, lt_ctx, TYPE):
    analysis = PickFrameAnalysis(dataset=default_raw, parameters={"x": 15, "y": 15})
    analysis.TYPE = TYPE
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)
    assert np.count_nonzero(results[0].raw_data) > 0


def test_roi_1(default_raw, lt_ctx):
    p = next(default_raw.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav().nav, dtype=bool)
    roi[0] = 1
    tiles = []
    for tile in p.get_tiles(dest_dtype="float32", roi=roi):
        print("tile:", tile)
        tiles.append(tile)
    assert len(tiles) == 1
    assert tiles[0].tile_slice.origin == (0, 0, 0)
    assert tuple(tiles[0].tile_slice.shape) == (1, 128, 128)


def test_roi_2(default_raw, lt_ctx):
    p = next(default_raw.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav(), dtype=bool)
    stackheight = p._get_stackheight(sig_shape=p.meta.shape.sig, dest_dtype=np.dtype("float32"))
    roi[0:stackheight + 2] = 1
    tiles = list(p.get_tiles(dest_dtype="float32", roi=roi))


def test_uint16_as_float32(uint16_raw, lt_ctx):
    p = next(uint16_raw.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav(), dtype=bool)
    stackheight = p._get_stackheight(sig_shape=p.meta.shape.sig, dest_dtype=np.dtype("float32"))
    roi[0:stackheight + 2] = 1
    tiles = list(p.get_tiles(dest_dtype="float32", roi=roi))


def test_macrotile_normal(lt_ctx, default_raw):
    ps = default_raw.get_partitions()
    _ = next(ps)
    p2 = next(ps)
    macrotile = p2.get_macrotile()
    assert macrotile.tile_slice.shape == p2.shape
    assert macrotile.tile_slice.origin[0] == p2._start_frame


def test_macrotile_roi_1(lt_ctx, default_raw):
    roi = np.zeros(default_raw.shape.nav, dtype=bool)
    roi[0, 5] = 1
    roi[0, 1] = 1
    p = next(default_raw.get_partitions())
    macrotile = p.get_macrotile(roi=roi)
    assert tuple(macrotile.tile_slice.shape) == (2, 128, 128)


def test_macrotile_roi_2(lt_ctx, default_raw):
    roi = np.zeros(default_raw.shape.nav, dtype=bool)
    # all ones are in the first partition, so we don't get any data in p2:
    roi[0, 5] = 1
    roi[0, 1] = 1
    ps = default_raw.get_partitions()
    _ = next(ps)
    p2 = next(ps)
    macrotile = p2.get_macrotile(roi=roi)
    assert tuple(macrotile.tile_slice.shape) == (0, 128, 128)


def test_macrotile_roi_3(lt_ctx, default_raw):
    roi = np.ones(default_raw.shape.nav, dtype=bool)
    ps = default_raw.get_partitions()
    _ = next(ps)
    p2 = next(ps)
    macrotile = p2.get_macrotile(roi=roi)
    assert tuple(macrotile.tile_slice.shape) == tuple(p2.shape)


def test_cache_key_json_serializable(default_raw):
    json.dumps(default_raw.get_cache_key())


def test_message_converter_direct():
    src = {
        "type": "RAW",
        "path": "p",
        "dtype": "d",
        "scan_size": [16, 16],
        "detector_size": [8, 8],
        "enable_direct": True,
    }
    converted = RAWDatasetParams().convert_to_python(src)
    assert converted == {
        "path": "p",
        "dtype": "d",
        "scan_size": [16, 16],
        "detector_size": [8, 8],
        "enable_direct": True,
    }


@pytest.mark.dist
def test_raw_on_workers(raw_on_workers, dist_ctx):
    # should not exist on the client node:
    assert not os.path.exists(raw_on_workers._path)
    res = dist_ctx.executor.run_each_host(os.path.exists, raw_on_workers._path)
    assert len(res) == 2
    assert all(res)


@pytest.mark.dist
def test_sum_on_dist(raw_on_workers, dist_ctx):
    print(dist_ctx.executor.run_each_host(lambda: os.system("hostname")))
    print(dist_ctx.executor.get_available_workers().group_by_host())
    print(dist_ctx.executor.get_available_workers())
    print(dist_ctx.executor.run_each_host(
        lambda: os.listdir(os.path.dirname(raw_on_workers._path))))
    analysis = dist_ctx.create_sum_analysis(dataset=raw_on_workers)
    results = dist_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)
