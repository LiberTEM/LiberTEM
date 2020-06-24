import os
import sys
import json
import pickle

import numpy as np
import pytest
import warnings
import sparse

from libertem.udf.sum import SumUDF
from libertem.udf.raw import PickUDF
from libertem.udf.masks import ApplyMasksUDF
from libertem.corrections import CorrectionSet
from libertem.corrections.detector import correct
from libertem.job.masks import ApplyMasksJob
from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.raw import PickFrameAnalysis
from libertem.io.dataset.raw import RAWDatasetParams
from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape


def test_simple_open(default_raw):
    assert tuple(default_raw.shape) == (16, 16, 128, 128)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No general sparse support on OS X")
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


@pytest.mark.with_numba
def test_read(default_raw):
    partitions = default_raw.get_partitions()
    p = next(partitions)
    # FIXME: partition shape can vary by number of cores
    # assert tuple(p.shape) == (2, 16, 128, 128)

    tileshape = Shape(
        (16,) + tuple(default_raw.shape.sig),
        sig_dims=default_raw.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_raw.shape,
    )

    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
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


def test_correction_default(default_raw, lt_ctx):
    ds = default_raw
    data = lt_ctx.run_udf(udf=PickUDF(), dataset=ds)

    gain = np.random.random(ds.shape.sig) + 1
    dark = np.random.random(ds.shape.sig) - 0.5
    exclude = [(np.random.randint(0, s), np.random.randint(0, s)) for s in tuple(ds.shape.sig)]

    exclude_coo = sparse.COO(coords=exclude, data=True, shape=ds.shape.sig)
    corrset = CorrectionSet(dark=dark, gain=gain, excluded_pixels=exclude_coo)

    def mask_factory():
        s = tuple(ds.shape.sig)
        return sparse.eye(np.prod(s)).reshape((-1, *s))

    # This one casts to float
    mask_res = lt_ctx.run_udf(udf=ApplyMasksUDF(mask_factory), dataset=ds, corrections=corrset)
    # This one uses native input data
    pick_res = lt_ctx.run_udf(udf=PickUDF(), dataset=ds, corrections=corrset)
    corrected = correct(
        buffer=data['intensity'].data.reshape(ds.shape),
        dark_image=dark,
        gain_map=gain,
        excluded_pixels=exclude,
        inplace=False
    )

    print(pick_res['intensity'].data.dtype)
    print(mask_res['intensity'].data.dtype)
    print(corrected.dtype)

    assert np.allclose(
        pick_res['intensity'].data.reshape(ds.shape),
        corrected
    )
    assert np.allclose(
        pick_res['intensity'].data.reshape(ds.shape),
        mask_res['intensity'].data.reshape(ds.shape),
    )


@pytest.mark.with_numba
def test_roi_1(default_raw, lt_ctx):
    p = next(default_raw.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav().nav, dtype=bool)
    roi[0] = 1
    tiles = []
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=Shape((1, 128, 128), sig_dims=2),
        dataset_shape=default_raw.shape,
    )
    for tile in p.get_tiles(tiling_scheme=tiling_scheme, dest_dtype="float32", roi=roi):
        print("tile:", tile)
        tiles.append(tile)
    assert len(tiles) == 1
    assert tiles[0].tile_slice.origin == (0, 0, 0)
    assert tuple(tiles[0].tile_slice.shape) == (1, 128, 128)


@pytest.mark.with_numba
def test_roi_2(default_raw, lt_ctx):
    p = next(default_raw.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav(), dtype=bool)
    stackheight = 4
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=Shape((stackheight, 128, 128), sig_dims=2),
        dataset_shape=default_raw.shape,
    )
    roi[0:stackheight + 2] = 1
    tiles = p.get_tiles(tiling_scheme=tiling_scheme, dest_dtype="float32", roi=roi)
    tiles = list(tiles)


def test_uint16_as_float32(uint16_raw, lt_ctx):
    p = next(uint16_raw.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav(), dtype=bool)

    stackheight = 4
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=Shape((stackheight, 128, 128), sig_dims=2),
        dataset_shape=uint16_raw.shape,
    )
    roi[0:stackheight + 2] = 1
    tiles = p.get_tiles(tiling_scheme=tiling_scheme, dest_dtype="float32", roi=roi)
    tiles = list(tiles)


def test_correction_uint16(uint16_raw, lt_ctx):
    ds = uint16_raw
    data = lt_ctx.run_udf(udf=PickUDF(), dataset=ds)

    gain = np.random.random(ds.shape.sig) + 1
    dark = np.random.random(ds.shape.sig) - 0.5
    exclude = [(np.random.randint(0, s), np.random.randint(0, s)) for s in tuple(ds.shape.sig)]

    exclude_coo = sparse.COO(coords=exclude, data=True, shape=ds.shape.sig)
    corrset = CorrectionSet(dark=dark, gain=gain, excluded_pixels=exclude_coo)

    def mask_factory():
        s = tuple(ds.shape.sig)
        return sparse.eye(np.prod(s)).reshape((-1, *s))

    # This one casts to float
    mask_res = lt_ctx.run_udf(udf=ApplyMasksUDF(mask_factory), dataset=ds, corrections=corrset)
    # This one uses native input data
    pick_res = lt_ctx.run_udf(udf=PickUDF(), dataset=ds, corrections=corrset)
    corrected = correct(
        buffer=data['intensity'].data.reshape(ds.shape),
        dark_image=dark,
        gain_map=gain,
        excluded_pixels=exclude,
        inplace=False
    )

    print(pick_res['intensity'].data.dtype)
    print(mask_res['intensity'].data.dtype)
    print(corrected.dtype)

    assert np.allclose(
        pick_res['intensity'].data.reshape(ds.shape),
        corrected
    )
    assert np.allclose(
        pick_res['intensity'].data.reshape(ds.shape),
        mask_res['intensity'].data.reshape(ds.shape),
    )


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

    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=p2.shape,
        dataset_shape=default_raw.shape,
    )
    p2._get_read_ranges(tiling_scheme, roi=None)
    p2._get_read_ranges(tiling_scheme, roi=roi)

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


def test_ctx_load_old(lt_ctx, default_raw):
    with warnings.catch_warnings(record=True) as w:
        lt_ctx.load(
            "raw",
            path=default_raw._path,
            scan_size=(16, 16),
            dtype="float32",
            detector_size_raw=(128, 128),
            crop_detector_to=(128, 128)
        )
        assert len(w) == 1
        assert issubclass(w[0].category, FutureWarning)


def test_missing_detector_size(lt_ctx, default_raw):
    with pytest.raises(TypeError) as e:
        lt_ctx.load(
            "raw",
            path=default_raw._path,
            scan_size=(16, 16),
            dtype="float32",
        )
    assert e.match("missing 1 required argument: 'detector_size'")


@pytest.mark.skipif(not sys.platform.startswith('linux'),
                    reason='Direct I/O only implemented on Linux')
def test_load_direct(lt_ctx, default_raw):
    ds_direct = lt_ctx.load(
        "raw",
        path=default_raw._path,
        scan_size=(16, 16),
        detector_size=(16, 16),
        dtype="float32",
        enable_direct=True,
    )
    analysis = lt_ctx.create_sum_analysis(dataset=ds_direct)
    lt_ctx.run(analysis)


@pytest.mark.skipif(sys.platform.startswith('linux'),
                    reason='No direct IO only on non-Linux')
def test_direct_io_enabled_non_linux(lt_ctx, default_raw):
    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "raw",
            path=default_raw._path,
            scan_size=(16, 16),
            detector_size=(16, 16),
            dtype="float32",
            enable_direct=True,
        )
    assert e.match("LiberTEM currently only supports Direct I/O on Linux")


def test_big_endian(big_endian_raw, lt_ctx):
    udf = SumUDF()
    lt_ctx.run_udf(udf=udf, dataset=big_endian_raw)


def test_correction_big_endian(big_endian_raw, lt_ctx):
    ds = big_endian_raw
    data = lt_ctx.run_udf(udf=PickUDF(), dataset=ds)

    gain = np.random.random(ds.shape.sig) + 1
    dark = np.random.random(ds.shape.sig) - 0.5
    exclude = [(np.random.randint(0, s), np.random.randint(0, s)) for s in tuple(ds.shape.sig)]

    exclude_coo = sparse.COO(coords=exclude, data=True, shape=ds.shape.sig)
    corrset = CorrectionSet(dark=dark, gain=gain, excluded_pixels=exclude_coo)

    def mask_factory():
        s = tuple(ds.shape.sig)
        return sparse.eye(np.prod(s)).reshape((-1, *s))

    # This one casts to float
    mask_res = lt_ctx.run_udf(udf=ApplyMasksUDF(mask_factory), dataset=ds, corrections=corrset)
    # This one uses native input data
    pick_res = lt_ctx.run_udf(udf=PickUDF(), dataset=ds, corrections=corrset)
    corrected = correct(
        buffer=data['intensity'].data.reshape(ds.shape),
        dark_image=dark,
        gain_map=gain,
        excluded_pixels=exclude,
        inplace=False
    )

    print(pick_res['intensity'].data.dtype)
    print(mask_res['intensity'].data.dtype)
    print(corrected.dtype)

    assert np.allclose(
        pick_res['intensity'].data.reshape(ds.shape),
        corrected
    )
    assert np.allclose(
        pick_res['intensity'].data.reshape(ds.shape),
        mask_res['intensity'].data.reshape(ds.shape),
    )


# TODO: test for dataset with more than 2 sig dims
