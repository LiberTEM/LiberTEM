import os
import sys
import json
import random

import numpy as np
import pytest
import cloudpickle

from libertem.udf.sum import SumUDF
from libertem.udf.raw import PickUDF

from libertem.analysis.raw import PickFrameAnalysis
from libertem.io.dataset.raw import RAWDatasetParams, RawFileDataSet
from libertem.io.dataset.base import (
    TilingScheme, BufferedBackend, MMapBackend, DirectBackend,
)
from libertem.common import Shape
from libertem.common.buffers import reshaped_view
from libertem.udf.sumsigudf import SumSigUDF

from utils import dataset_correction_verification, ValidationUDF, roi_as_sparse


@pytest.fixture
def raw_dataset_8x8x8x8(lt_ctx, raw_data_8x8x8x8_path):
    ds = RawFileDataSet(
        path=raw_data_8x8x8x8_path,
        nav_shape=(8, 8),
        sig_shape=(8, 8),
        dtype="float32",
        num_partitions=4,
    )
    ds = ds.initialize(lt_ctx.executor)

    return ds


def test_simple_open(default_raw):
    assert tuple(default_raw.shape) == (16, 16, 128, 128)


@pytest.mark.skipif(sys.platform.startswith("darwin"), reason="No general sparse support on OS X")
def test_large_pick(large_raw, lt_ctx):
    y, x = large_raw.shape.nav
    dy, dx = large_raw.shape.sig
    analysis = lt_ctx.create_pick_analysis(large_raw, y=y-1, x=x-1)
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
    for _ in tiles:
        pass


def test_scheme_too_large(default_raw):
    partitions = default_raw.get_partitions()
    p = next(partitions)
    depth = p.shape[0]

    # we make a tileshape that is too large for the partition here:
    tileshape = Shape(
        (depth + 1,) + tuple(default_raw.shape.sig),
        sig_dims=default_raw.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_raw.shape,
    )

    # tile shape is clamped to partition shape:
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == tuple((depth,) + default_raw.shape.sig)
    for _ in tiles:
        pass


def test_comparison(default_raw, default_raw_data, lt_ctx_fast):
    udf = ValidationUDF(
        reference=reshaped_view(default_raw_data, (-1, *tuple(default_raw.shape.sig)))
    )
    lt_ctx_fast.run_udf(udf=udf, dataset=default_raw)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_comparison_roi(default_raw, default_raw_data, lt_ctx_fast, as_sparse):
    roi = np.random.choice(
        [True, False],
        size=tuple(default_raw.shape.nav),
        p=[0.5, 0.5]
    )
    udf = ValidationUDF(reference=default_raw_data[roi])
    if as_sparse:
        roi = roi_as_sparse(roi)
    lt_ctx_fast.run_udf(udf=udf, dataset=default_raw, roi=roi)


def test_pickle_is_small(default_raw):
    pickled = cloudpickle.dumps(default_raw)
    cloudpickle.loads(pickled)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 2 * 1024


def test_apply_mask_analysis(default_raw, lt_ctx):
    mask = np.ones((128, 128))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_raw)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (16, 16)


def test_sum_analysis(default_raw, lt_ctx):
    analysis = lt_ctx.create_sum_analysis(dataset=default_raw)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)


def test_pick_analysis(default_raw, lt_ctx):
    analysis = PickFrameAnalysis(dataset=default_raw, parameters={"x": 15, "y": 15})
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)
    assert np.count_nonzero(results[0].raw_data) > 0


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_correction_default(default_raw, lt_ctx, with_roi, as_sparse):
    ds = default_raw

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None
    if with_roi and as_sparse:
        roi = roi_as_sparse(roi)
    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


@pytest.mark.with_numba
@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_roi_1(default_raw, lt_ctx, as_sparse):
    p = next(default_raw.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav().nav, dtype=bool)
    roi[0] = 1
    tiles = []
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=Shape((1, 128, 128), sig_dims=2),
        dataset_shape=default_raw.shape,
    )
    if as_sparse:
        roi = roi_as_sparse(roi)
    for tile in p.get_tiles(tiling_scheme=tiling_scheme, dest_dtype="float32", roi=roi):
        print("tile:", tile)
        tiles.append(tile)
    assert len(tiles) == 1
    assert tiles[0].tile_slice.origin == (0, 0, 0)
    assert tuple(tiles[0].tile_slice.shape) == (1, 128, 128)


@pytest.mark.with_numba
@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_roi_2(default_raw, lt_ctx, as_sparse):
    p = next(default_raw.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav(), dtype=bool)
    stackheight = 4
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=Shape((stackheight, 128, 128), sig_dims=2),
        dataset_shape=default_raw.shape,
    )
    roi[0:stackheight + 2] = 1
    if as_sparse:
        roi = roi_as_sparse(roi)
    tiles = p.get_tiles(tiling_scheme=tiling_scheme, dest_dtype="float32", roi=roi)
    tiles = list(tiles)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_uint16_as_float32(uint16_raw, lt_ctx, as_sparse):
    p = next(uint16_raw.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav(), dtype=bool)

    stackheight = 4
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=Shape((stackheight, 128, 128), sig_dims=2),
        dataset_shape=uint16_raw.shape,
    )
    roi[0:stackheight + 2] = 1
    if as_sparse:
        roi = roi_as_sparse(roi)
    tiles = p.get_tiles(tiling_scheme=tiling_scheme, dest_dtype="float32", roi=roi)
    tiles = list(tiles)


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_correction_uint16(uint16_raw, lt_ctx, with_roi, as_sparse):
    ds = uint16_raw

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None
    if with_roi and as_sparse:
        roi = roi_as_sparse(roi)
    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


def test_macrotile_normal(lt_ctx, default_raw):
    ps = default_raw.get_partitions()
    _ = next(ps)
    p2 = next(ps)
    macrotile = p2.get_macrotile()
    assert macrotile.tile_slice.shape == p2.shape
    assert macrotile.tile_slice.origin[0] == p2._start_frame


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_macrotile_roi_1(lt_ctx, default_raw, as_sparse):
    roi = np.zeros(default_raw.shape.nav, dtype=bool)
    roi[0, 5] = 1
    roi[0, 1] = 1
    p = next(default_raw.get_partitions())
    if as_sparse:
        roi = roi_as_sparse(roi)
    macrotile = p.get_macrotile(roi=roi)
    assert tuple(macrotile.tile_slice.shape) == (2, 128, 128)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_macrotile_roi_2(lt_ctx, default_raw, as_sparse):
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
    if as_sparse:
        roi = roi_as_sparse(roi)
    p2._get_read_ranges(tiling_scheme, roi=None)
    p2._get_read_ranges(tiling_scheme, roi=roi)

    macrotile = p2.get_macrotile(roi=roi)
    assert tuple(macrotile.tile_slice.shape) == (0, 128, 128)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_macrotile_roi_3(lt_ctx, default_raw, as_sparse):
    roi = np.ones(default_raw.shape.nav, dtype=bool)
    ps = default_raw.get_partitions()
    _ = next(ps)
    p2 = next(ps)
    if as_sparse:
        roi = roi_as_sparse(roi)
    macrotile = p2.get_macrotile(roi=roi)
    assert tuple(macrotile.tile_slice.shape) == tuple(p2.shape)


def test_cache_key_json_serializable(default_raw):
    json.dumps(default_raw.get_cache_key())


def test_message_converter_direct():
    src = {
        "type": "RAW",
        "path": "p",
        "dtype": "d",
        "nav_shape": [16, 16],
        "sig_shape": [8, 8],
        "sync_offset": 0,
    }
    converted = RAWDatasetParams().convert_to_python(src)
    assert converted == {
        "path": "p",
        "dtype": "d",
        "nav_shape": [16, 16],
        "sig_shape": [8, 8],
        "sync_offset": 0,
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


def test_ctx_load_old(lt_ctx, default_raw, recwarn):
    lt_ctx.load(
        "raw",
        path=default_raw._path,
        nav_shape=(16, 16),
        dtype="float32",
        detector_size_raw=(128, 128),
        crop_detector_to=(128, 128)
    )
    assert len(recwarn) == 2
    w0 = recwarn.pop(FutureWarning)
    assert issubclass(w0.category, FutureWarning)
    w1 = recwarn.pop(FutureWarning)
    assert issubclass(w1.category, FutureWarning)


def test_missing_sig_shape(lt_ctx, default_raw):
    with pytest.raises(TypeError) as e:
        lt_ctx.load(
            "raw",
            path=default_raw._path,
            nav_shape=(16, 16),
            dtype="float32",
        )
    assert e.match("missing 1 required argument: 'sig_shape'")


@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="No support for direct I/O on Mac OS X"
)
def test_load_direct(lt_ctx, default_raw):
    ds_direct = lt_ctx.load(
        "raw",
        path=default_raw._path,
        nav_shape=(16, 16),
        sig_shape=(16, 16),
        dtype="float32",
        io_backend=DirectBackend(),
    )
    analysis = lt_ctx.create_sum_analysis(dataset=ds_direct)
    lt_ctx.run(analysis)


@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="No support for direct I/O on Mac OS X"
)
def test_load_legacy_direct(lt_ctx, default_raw):
    with pytest.warns(FutureWarning):
        ds_direct = lt_ctx.load(
            "raw",
            path=default_raw._path,
            nav_shape=(16, 16),
            sig_shape=(16, 16),
            dtype="float32",
            enable_direct=True,
        )
    analysis = lt_ctx.create_sum_analysis(dataset=ds_direct)
    lt_ctx.run(analysis)


@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="No support for direct I/O on Mac OS X"
)
def test_compare_direct_to_mmap(lt_ctx, default_raw, direct_raw):
    y = random.choice(range(default_raw.shape.nav[0]))
    x = random.choice(range(default_raw.shape.nav[1]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_raw,
        x=x, y=y,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=direct_raw,
        x=x, y=y,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


def test_big_endian(big_endian_raw, lt_ctx):
    udf = SumUDF()
    lt_ctx.run_udf(udf=udf, dataset=big_endian_raw)


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_correction_big_endian(big_endian_raw, lt_ctx, with_roi, as_sparse):
    ds = big_endian_raw
    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
        if as_sparse:
            roi = roi_as_sparse(roi)
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


@pytest.mark.parametrize(
    "io_backend", (
        BufferedBackend(),
        MMapBackend(),
    ),
)
def test_positive_sync_offset(lt_ctx, raw_dataset_8x8x8x8, raw_data_8x8x8x8_path, io_backend):
    udf = SumSigUDF()
    sync_offset = 2

    ds_with_offset = RawFileDataSet(
        path=raw_data_8x8x8x8_path,
        nav_shape=(8, 8),
        sig_shape=(8, 8),
        dtype="float32",
        sync_offset=sync_offset,
        io_backend=io_backend,
        num_partitions=4,
    )
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == 2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (4,) + tuple(ds_with_offset.shape.sig),
        sig_dims=ds_with_offset.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds_with_offset.shape,
    )

    tiles = p0.get_tiles(tiling_scheme)
    t0 = next(tiles)
    assert tuple(t0.tile_slice.origin) == (0, 0, 0)
    for _ in tiles:
        pass

    for p in ds_with_offset.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p.slice.origin == (48, 0, 0)
    assert p.slice.shape[0] == 16

    result = lt_ctx.run_udf(dataset=raw_dataset_8x8x8x8, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
        :ds_with_offset._meta.image_count - sync_offset
    ]

    assert np.allclose(result, result_with_offset)


@pytest.mark.parametrize(
    "io_backend", (
        BufferedBackend(),
        MMapBackend(),
    ),
)
def test_negative_sync_offset(lt_ctx, raw_dataset_8x8x8x8, raw_data_8x8x8x8_path, io_backend):
    udf = SumSigUDF()
    sync_offset = -2

    ds_with_offset = RawFileDataSet(
        path=raw_data_8x8x8x8_path,
        nav_shape=(8, 8),
        sig_shape=(8, 8),
        dtype="float32",
        sync_offset=sync_offset,
        io_backend=io_backend,
        num_partitions=4,
    )
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == -2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (4,) + tuple(ds_with_offset.shape.sig),
        sig_dims=ds_with_offset.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds_with_offset.shape,
    )

    tiles = p0.get_tiles(tiling_scheme)
    t0 = next(tiles)
    assert tuple(t0.tile_slice.origin) == (2, 0, 0)
    for _ in tiles:
        pass

    for p in ds_with_offset.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p.slice.origin == (48, 0, 0)
    assert p.slice.shape[0] == 16

    result = lt_ctx.run_udf(dataset=raw_dataset_8x8x8x8, udf=udf)
    result = result['intensity'].raw_data[:raw_dataset_8x8x8x8._meta.image_count - abs(sync_offset)]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


@pytest.mark.parametrize(
    "io_backend", (
        BufferedBackend(),
        MMapBackend(),
    ),
)
def test_missing_frames(lt_ctx, raw_data_8x8x8x8_path, io_backend):
    ds = RawFileDataSet(
        path=raw_data_8x8x8x8_path,
        nav_shape=(10, 8),
        sig_shape=(8, 8),
        dtype="float32",
        io_backend=io_backend,
        num_partitions=4,
    )
    ds = ds.initialize(lt_ctx.executor)

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p._start_frame == 60
    assert p._num_frames == 20
    assert p.slice.origin == (60, 0, 0)
    assert p.slice.shape[0] == 20
    assert t.tile_slice.origin == (60, 0, 0)
    assert t.tile_slice.shape[0] == 4


@pytest.mark.parametrize(
    "io_backend", (
        BufferedBackend(),
        MMapBackend(),
    ),
)
def test_too_many_frames(lt_ctx, raw_data_8x8x8x8_path, io_backend):
    ds = RawFileDataSet(
        path=raw_data_8x8x8x8_path,
        nav_shape=(6, 8),
        sig_shape=(8, 8),
        dtype="float32",
        io_backend=io_backend,
        num_partitions=4,
    )
    ds = ds.initialize(lt_ctx.executor)

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass


def test_offset_smaller_than_image_count(lt_ctx, raw_data_8x8x8x8_path):
    sync_offset = -70

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "raw",
            path=raw_data_8x8x8x8_path,
            nav_shape=(8, 8),
            sig_shape=(8, 8),
            dtype="float32",
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-64, 64\), which is \(-image_count, image_count\)"
    )


def test_offset_greater_than_image_count(lt_ctx, raw_data_8x8x8x8_path):
    sync_offset = 70

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "raw",
            path=raw_data_8x8x8x8_path,
            nav_shape=(8, 8),
            sig_shape=(8, 8),
            dtype="float32",
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-64, 64\), which is \(-image_count, image_count\)"
    )


@pytest.mark.parametrize(
    "io_backend", (
        BufferedBackend(),
        MMapBackend(),
    ),
)
def test_reshape_nav(lt_ctx, raw_dataset_8x8x8x8, raw_data_8x8x8x8_path, io_backend):
    udf = SumSigUDF()

    ds_with_1d_nav = lt_ctx.load(
        "raw",
        path=raw_data_8x8x8x8_path,
        nav_shape=(64,),
        sig_shape=(8, 8),
        dtype="float32",
        io_backend=io_backend,
    )
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    result_with_2d_nav = lt_ctx.run_udf(dataset=raw_dataset_8x8x8x8, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load(
        "raw",
        path=raw_data_8x8x8x8_path,
        nav_shape=(2, 4, 8),
        sig_shape=(8, 8),
        dtype="float32",
    )
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


def test_too_large_sig_shape(lt_ctx, raw_data_8x8x8x8_path):
    nav_shape = (8, 8)
    sig_shape = (65, 65)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "raw",
            path=raw_data_8x8x8x8_path,
            nav_shape=nav_shape,
            sig_shape=sig_shape,
            dtype="float32",
        )
    assert e.match(
        r"sig_shape must be less than size: 4096"
    )


def test_different_sig_shape(lt_ctx, raw_data_8x8x8x8_path):
    nav_shape = (8, 8)
    sig_shape = (4, 4)

    ds = lt_ctx.load(
        "raw",
        path=raw_data_8x8x8x8_path,
        nav_shape=nav_shape,
        sig_shape=sig_shape,
        dtype="float32",
    )

    assert ds._meta.image_count == 256


@pytest.mark.parametrize(
    "io_backend", (
        BufferedBackend(),
        MMapBackend(),
    ),
)
def test_extra_data_at_the_end(lt_ctx, raw_data_8x8x8x8_path, io_backend):
    """
    If there is extra data at the end of the file, make sure it is cut off
    """
    nav_shape = (8, 8)
    sig_shape = (3, 3)

    ds = lt_ctx.load(
        "raw",
        path=raw_data_8x8x8x8_path,
        nav_shape=nav_shape,
        sig_shape=sig_shape,
        dtype="float32",
        io_backend=io_backend,
    )

    assert ds._meta.image_count == 455


def test_scan_size_deprecation(lt_ctx, raw_data_8x8x8x8_path):
    scan_size = (2, 2)

    with pytest.warns(FutureWarning):
        ds = lt_ctx.load(
            "raw",
            path=raw_data_8x8x8x8_path,
            scan_size=scan_size,
            sig_shape=(8, 8),
            dtype="float32",
        )
    assert tuple(ds.shape) == (2, 2, 8, 8)


def test_detector_size_deprecation(lt_ctx, raw_data_8x8x8x8_path):
    detector_size = (8, 8)

    with pytest.warns(FutureWarning):
        ds = lt_ctx.load(
            "raw",
            path=raw_data_8x8x8x8_path,
            nav_shape=(8, 8),
            detector_size=detector_size,
            dtype="float32",
        )
    assert tuple(ds.shape) == (8, 8, 8, 8)


def test_compare_backends(lt_ctx, default_raw, buffered_raw):
    y = random.choice(range(default_raw.shape.nav[0]))
    x = random.choice(range(default_raw.shape.nav[1]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_raw,
        x=x, y=y,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=buffered_raw,
        x=x, y=y,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_compare_backends_sparse(lt_ctx, default_raw, buffered_raw, as_sparse):
    roi = np.zeros(default_raw.shape.nav, dtype=bool).reshape((-1,))
    roi[0] = True
    roi[1] = True
    roi[-1] = True
    if as_sparse:
        roi = roi_as_sparse(roi)
    mm_f0 = lt_ctx.run_udf(dataset=default_raw, udf=PickUDF(), roi=roi)['intensity']
    buffered_f0 = lt_ctx.run_udf(dataset=buffered_raw, udf=PickUDF(), roi=roi)['intensity']

    assert np.allclose(mm_f0, buffered_f0)


# TODO: test for dataset with more than 2 sig dims

def test_diagnostics(default_raw):
    assert {"name": "dtype", "value": "float32"} in default_raw.get_diagnostics()


def test_bad_params(ds_params_tester, standard_bad_ds_params, raw_dataset_8x8x8x8):
    args = ("raw", raw_dataset_8x8x8x8._path)
    for params in standard_bad_ds_params:
        params['dtype'] = raw_dataset_8x8x8x8.meta.raw_dtype
        if 'nav_shape' not in params:
            params['nav_shape'] = raw_dataset_8x8x8x8.meta.shape.nav
        if 'sig_shape' not in params:
            params['sig_shape'] = raw_dataset_8x8x8x8.meta.shape.sig
        ds_params_tester(*args, **params)
