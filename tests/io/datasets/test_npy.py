import os
import sys
import json

import numpy as np
import pytest
import cloudpickle

import libertem.api as lt

from libertem.analysis.raw import PickFrameAnalysis
from libertem.io.dataset.npy import NPYDatasetParams, NPYDataSet, read_npy_info
from libertem.io.dataset.base import (
    TilingScheme, BufferedBackend, MMapBackend, DirectBackend, DataSetException
)
from libertem.common import Shape
from libertem.common.math import prod
from libertem.common.buffers import reshaped_view
from libertem.udf.sumsigudf import SumSigUDF

from utils import dataset_correction_verification, ValidationUDF, roi_as_sparse


@pytest.mark.parametrize(
    "repeat", [*range(50)]
)
def test_read_npy_info(npy_random_array, repeat):
    npy_filepath, array = npy_random_array

    info = read_npy_info(npy_filepath)
    assert info.dtype == array.dtype
    assert info.shape == array.shape
    assert info.count == array.size
    # compute the header offset from the other direction !
    offset = os.stat(npy_filepath).st_size - array.nbytes
    assert info.offset == offset


def test_read_npy_info_fortran(npy_fortran_array):
    npy_filepath, array = npy_fortran_array
    assert array.flags.f_contiguous
    with pytest.raises(DataSetException):
        read_npy_info(npy_filepath)


def test_repr(default_npy):
    assert 'NPYDataSet' in repr(default_npy)


def test_get_fileset(default_npy):
    fileset = default_npy._get_fileset()
    assert len(fileset) == 1
    assert fileset[0]._path == default_npy._path


def test_detect_params(default_npy, default_raw_data, lt_ctx):
    params = NPYDataSet.detect_params(default_npy._path, lt_ctx.executor)
    assert (params['parameters']['nav_shape']
        + params['parameters']['sig_shape'])\
        == default_raw_data.shape


def test_detect_params_fail(lt_ctx):
    params = NPYDataSet.detect_params(555, lt_ctx.executor)
    assert not params


def test_apply_mask_analysis(default_npy, lt_ctx):
    mask = np.ones((128, 128))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_npy)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (16, 16)


def test_sum_analysis(default_npy, lt_ctx):
    analysis = lt_ctx.create_sum_analysis(dataset=default_npy)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)


def test_pick_analysis(default_npy, lt_ctx):
    analysis = PickFrameAnalysis(dataset=default_npy, parameters={"x": 15, "y": 15})
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (128, 128)
    assert np.count_nonzero(results[0].raw_data) > 0


def test_comparison(default_npy, default_raw_data, lt_ctx_fast):
    udf = ValidationUDF(
        reference=reshaped_view(default_raw_data, (-1, *tuple(default_npy.shape.sig)))
    )
    lt_ctx_fast.run_udf(udf=udf, dataset=default_npy)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_comparison_roi(default_npy, default_raw_data, lt_ctx_fast, as_sparse):
    roi = np.random.choice(
        [True, False],
        size=tuple(default_npy.shape.nav),
        p=[0.5, 0.5]
    )
    ref_data = default_raw_data[roi]
    if as_sparse:
        roi = roi_as_sparse(roi)
    udf = ValidationUDF(reference=ref_data)
    lt_ctx_fast.run_udf(udf=udf, dataset=default_npy, roi=roi)


def test_pickle_is_small(default_npy):
    pickled = cloudpickle.dumps(default_npy)
    cloudpickle.loads(pickled)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 2 * 1024


def test_cache_key_json_serializable(default_npy):
    json.dumps(default_npy.get_cache_key())


def test_message_converter_direct():
    src = {
        "type": "NPY",
        "path": "p",
        "nav_shape": [16, 16],
        "sig_shape": [8, 8],
    }
    converted = NPYDatasetParams().convert_to_python(src)
    assert converted == {
        "path": "p",
        "nav_shape": (16, 16),
        "sig_shape": (8, 8),
    }


def test_bad_sig_dims(lt_ctx):
    with pytest.raises(DataSetException):
        lt_ctx.load(
            "npy",
            path='anything.npy',
            sig_shape=(1, 2, 3),
            sig_dims=1,
        )


def test_bad_sig_dims2(lt_ctx):
    with pytest.raises(DataSetException):
        lt_ctx.load(
            "npy",
            path='anything.npy',
            sig_dims=None,
        )


def test_auto_sig_dims(lt_ctx):
    ds = NPYDataSet(path='anything.npy',
                    sig_shape=(1, 2, 3),
                    sig_dims=None)
    assert ds._sig_dims == 3


def test_shape_arg_smallernav(default_npy, default_npy_filepath, default_raw_data, lt_ctx):
    nav_shape, sig_shape = default_raw_data.shape[:2], default_raw_data.shape[2:]
    smaller_nav_shape = tuple(n - 1 for n in nav_shape)
    ds = lt_ctx.load(
            "npy",
            path=default_npy_filepath,
            sig_dims=2,
            nav_shape=smaller_nav_shape,
        )
    assert tuple(ds.shape.sig) == sig_shape
    assert tuple(ds.shape.nav) == smaller_nav_shape
    lt_ctx: lt.Context
    result = lt_ctx.map(ds, lambda x: True)
    assert result.data.shape == smaller_nav_shape


def test_shape_arg_flatnav(default_npy, default_npy_filepath, default_raw_data, lt_ctx):
    nav_shape, sig_shape = default_raw_data.shape[:2], default_raw_data.shape[2:]
    flat_nav_shape = (prod(nav_shape),)
    ds = lt_ctx.load(
            "npy",
            path=default_npy_filepath,
            sig_dims=2,
            nav_shape=flat_nav_shape,
        )
    assert tuple(ds.shape.sig) == sig_shape
    assert tuple(ds.shape.nav) == flat_nav_shape
    lt_ctx: lt.Context
    result = lt_ctx.map(ds, lambda x: True)
    assert result.data.shape == flat_nav_shape


@pytest.mark.with_numba
@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_roi_1(default_npy, lt_ctx, as_sparse):
    p = next(default_npy.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav().nav, dtype=bool)
    roi[0] = 1
    tiles = []
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=Shape((1, 128, 128), sig_dims=2),
        dataset_shape=default_npy.shape,
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
def test_roi_2(default_npy, lt_ctx, as_sparse):
    p = next(default_npy.get_partitions())
    roi = np.zeros(p.meta.shape.flatten_nav(), dtype=bool)
    stackheight = 4
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=Shape((stackheight, 128, 128), sig_dims=2),
        dataset_shape=default_npy.shape,
    )
    roi[0:stackheight + 2] = 1
    if as_sparse:
        roi = roi_as_sparse(roi)
    tiles = p.get_tiles(tiling_scheme=tiling_scheme, dest_dtype="float32", roi=roi)
    tiles = list(tiles)


@pytest.mark.parametrize(
    "io_backend", (
        BufferedBackend(),
        MMapBackend(),
    ),
)
def test_positive_sync_offset(lt_ctx, npy_8x8x8x8_ds, npy_8x8x8x8_path, io_backend):
    udf = SumSigUDF()
    sync_offset = 2

    ds_with_offset = NPYDataSet(
        path=npy_8x8x8x8_path,
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

    result = lt_ctx.run_udf(dataset=npy_8x8x8x8_ds, udf=udf)
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
def test_negative_sync_offset(lt_ctx, npy_8x8x8x8_ds, npy_8x8x8x8_path, io_backend):
    udf = SumSigUDF()
    sync_offset = -2

    ds_with_offset = NPYDataSet(
        path=npy_8x8x8x8_path,
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

    result = lt_ctx.run_udf(dataset=npy_8x8x8x8_ds, udf=udf)
    result = result['intensity'].raw_data[:npy_8x8x8x8_ds._meta.image_count - abs(sync_offset)]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="No support for direct I/O on Mac OS X"
)
def test_load_direct(lt_ctx, npy_8x8x8x8_path):
    ds_direct = lt_ctx.load(
        "npy",
        path=npy_8x8x8x8_path,
        io_backend=DirectBackend(),
    )
    analysis = lt_ctx.create_sum_analysis(dataset=ds_direct)
    lt_ctx.run(analysis)


def test_no_num_partitions(lt_ctx, npy_8x8x8x8_path):
    ds = lt_ctx.load(
        "npy",
        path=npy_8x8x8x8_path,
    )
    lt_ctx.run_udf(dataset=ds, udf=SumSigUDF())


@pytest.mark.parametrize(
    "io_backend", (
        BufferedBackend(),
        MMapBackend(),
    ),
)
def test_reshape_nav(lt_ctx, npy_8x8x8x8_ds, npy_8x8x8x8_path, io_backend):
    udf = SumSigUDF()

    ds_with_1d_nav = lt_ctx.load(
        "npy",
        path=npy_8x8x8x8_path,
        nav_shape=(64,),
        io_backend=io_backend,
    )
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    result_with_2d_nav = lt_ctx.run_udf(dataset=npy_8x8x8x8_ds, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load(
        "npy",
        path=npy_8x8x8x8_path,
        nav_shape=(2, 4, 8),
    )
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


def test_different_sig_shape(lt_ctx, npy_8x8x8x8_path):
    sig_shape = (4, 4)

    ds = lt_ctx.load(
        "npy",
        path=npy_8x8x8x8_path,
        sig_shape=sig_shape,
    )

    assert ds._meta.image_count == 256


@pytest.mark.parametrize(
    "io_backend", (
        BufferedBackend(),
        MMapBackend(),
    ),
)
def test_extra_data_at_the_end(lt_ctx, npy_8x8x8x8_path, io_backend):
    """
    If there is extra data at the end of the file, make sure it is cut off
    """
    sig_shape = (3, 3)

    ds = lt_ctx.load(
        "npy",
        path=npy_8x8x8x8_path,
        sig_shape=sig_shape,
        io_backend=io_backend,
    )

    assert ds._meta.image_count == 455


def test_offset_greater_than_image_count(lt_ctx, npy_8x8x8x8_path):
    sync_offset = 70

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "npy",
            path=npy_8x8x8x8_path,
            sync_offset=sync_offset,
        )
    assert e.match(
        r"offset should be in \(-64, 64\), which is \(-image_count, image_count\)"
    )


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_correction_default(default_npy, lt_ctx, with_roi):
    ds = default_npy

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


def test_diagnostics(default_npy):
    assert {"name": "dtype", "value": "float32"} in default_npy.get_diagnostics()


def test_bad_params(ds_params_tester, standard_bad_ds_params, default_npy):
    args = ("npy", default_npy._path)
    for params in standard_bad_ds_params:
        ds_params_tester(*args, **params)
