import os

from scipy.sparse import csr_matrix
import numpy as np
import pytest
from numpy.testing import assert_allclose

from libertem.io.dataset import raw_csr
from libertem.io.dataset.raw_csr import (
    read_tiles_straight, read_tiles_with_roi, CSRTriple, RawCSRDataSet
)
from sparseconverter import NUMPY, for_backend, SCIPY_CSR
from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape, Slice
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.stddev import StdDevUDF
from libertem.common.math import prod
from libertem.web.dataset import prime_numba_cache

from utils import _mk_random, get_testdata_path

RAW_CSR_TESTDATA_PATH = os.path.join(get_testdata_path(), 'raw_csr', 'sparse.toml')
HAVE_CSR_TESTDATA = os.path.exists(RAW_CSR_TESTDATA_PATH)


def test_get_tiles_straight():
    data = _mk_random((13, 17, 24, 19), array_backend=NUMPY)
    data_flat = data.reshape((13*17, 24*19))

    sig_shape = (24, 19)
    tileshape = Shape((11, ) + sig_shape, sig_dims=2)
    dataset_shape = Shape(data.shape, sig_dims=2)

    orig = for_backend(data_flat, SCIPY_CSR)

    orig_triple = CSRTriple(indptr=orig.indptr, indices=orig.indices, data=orig.data)
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=dataset_shape,
        intent='tile'
    )
    partition_slice = Slice(origin=(3, 0, 0), shape=Shape((51, ) + sig_shape, sig_dims=2))

    ref = for_backend(np.sum(data_flat[3:3+51], axis=0), NUMPY)

    res = np.zeros((1, data_flat.shape[1]), dtype=data.dtype)

    for tile in read_tiles_straight(
            triple=orig_triple,
            partition_slice=partition_slice,
            tiling_scheme=tiling_scheme,
            dest_dtype=res.dtype,
            sync_offset=0
    ):
        res += np.sum(tile.data, axis=0)
        assert tile.shape[0] <= tuple(tileshape)[0]
        assert isinstance(tile.data, csr_matrix)

    assert np.allclose(ref, res)


def test_get_tiles_simple():
    data = np.array((
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
    ))
    data_flat = data.reshape((4, 3))

    sig_shape = (3, )
    tileshape = Shape((1, ) + sig_shape, sig_dims=1)
    dataset_shape = Shape(data.shape, sig_dims=1)

    orig = for_backend(data_flat, SCIPY_CSR)

    orig_triple = CSRTriple(indptr=orig.indptr, indices=orig.indices, data=orig.data)
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=dataset_shape,
        intent='tile'
    )
    partition_slice = Slice(origin=(1, 0), shape=Shape((3, ) + sig_shape, sig_dims=1))

    ref = for_backend(np.sum(data_flat[1:], axis=0), NUMPY)

    res = np.zeros((1, data_flat.shape[1]), dtype=data.dtype)

    for tile in read_tiles_straight(
            triple=orig_triple,
            partition_slice=partition_slice,
            tiling_scheme=tiling_scheme,
            dest_dtype=res.dtype,
            sync_offset=0
    ):
        res += np.sum(tile.data, axis=0)
        assert tile.shape[0] <= tuple(tileshape)[0]
        assert isinstance(tile.data, csr_matrix)

    assert np.allclose(ref, res)


@pytest.mark.with_numba
def test_get_tiles_simple_roi():
    data = np.array((
        [1, 1, 1],
        [2, 2, 2],
        [3, 3, 3],
        [4, 4, 4],
    ))
    data_flat = data.reshape((4, 3))

    sig_shape = (3, )
    tileshape = Shape((1, ) + sig_shape, sig_dims=1)
    dataset_shape = Shape(data.shape, sig_dims=1)

    orig = for_backend(data_flat, SCIPY_CSR)

    orig_triple = CSRTriple(indptr=orig.indptr, indices=orig.indices, data=orig.data)
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=dataset_shape,
        intent='tile'
    )
    partition_slice = Slice(origin=(1, 0), shape=Shape((3, ) + sig_shape, sig_dims=1))
    roi = np.ones(data.shape[0], dtype=bool)
    roi[0:2] = 0
    roi[-1] = 0

    ref = for_backend(np.sum(data_flat[2:-1], axis=0), NUMPY)

    res = np.zeros((1, data_flat.shape[1]), dtype=data.dtype)

    for tile in read_tiles_with_roi(
            triple=orig_triple,
            partition_slice=partition_slice,
            tiling_scheme=tiling_scheme,
            roi=roi,
            dest_dtype=res.dtype,
            sync_offset=0
    ):
        res += np.sum(tile.data, axis=0)
        assert tile.shape[0] <= tuple(tileshape)[0]
        assert isinstance(tile.data, csr_matrix)

    assert np.allclose(ref, res)


def test_get_tiles_roi():
    data = _mk_random((13, 17, 24, 19), array_backend=NUMPY)
    data_flat = data.reshape((13*17, 24*19))

    sig_shape = (24, 19)
    tileshape = Shape((11, ) + sig_shape, sig_dims=2)
    dataset_shape = Shape(data.shape, sig_dims=2)

    orig = for_backend(data_flat, SCIPY_CSR)

    orig_triple = CSRTriple(indptr=orig.indptr, indices=orig.indices, data=orig.data)
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=dataset_shape,
        intent='tile'
    )
    roi = np.random.choice([True, False], data_flat.shape[0])
    partition_slice = Slice(origin=(3, 0, 0), shape=Shape((51, ) + sig_shape, sig_dims=2))
    roi_slice = partition_slice.get(roi, nav_only=True)

    ref = for_backend(np.sum(data_flat[3:3+51][roi_slice], axis=0), NUMPY)

    res = np.zeros((1, data_flat.shape[1]), dtype=data.dtype)
    total_size = 0
    for tile in read_tiles_with_roi(
            triple=orig_triple,
            partition_slice=partition_slice,
            tiling_scheme=tiling_scheme,
            roi=roi,
            dest_dtype=res.dtype,
            sync_offset=0
    ):
        res += np.sum(tile.data, axis=0)

        total_size += tile.data.shape[0]
        assert tile.shape[0] <= tuple(tileshape)[0]
        assert isinstance(tile.data, csr_matrix)
    assert total_size == np.count_nonzero(roi_slice)

    assert np.allclose(ref, res)


@pytest.mark.parametrize(
    'endian', ('native', 'big')
)
def test_raw_csr_ds_sum(
        endian, raw_csr_generated, mock_sparse_data,
        raw_csr_generated_bigendian, lt_ctx):
    if endian == 'native':
        _, data_flat = mock_sparse_data
        ds = raw_csr_generated
    elif endian == 'big':
        _, data_flat = mock_sparse_data
        ds = raw_csr_generated_bigendian
        assert ds.dtype == np.dtype('>i4')
    else:
        raise ValueError()
    udf = SumUDF()
    res = lt_ctx.run_udf(udf=udf, dataset=ds)
    ref = for_backend(np.sum(data_flat, axis=0), NUMPY)
    assert np.allclose(ref, res['intensity'].data.reshape((-1,)))


@pytest.mark.parametrize(
    'endian', ('native', 'big')
)
def test_raw_csr_ds_sum_roi(
        endian, raw_csr_generated, mock_sparse_data,
        raw_csr_generated_bigendian, lt_ctx):
    if endian == 'native':
        orig, data_flat = mock_sparse_data
        ds = raw_csr_generated
    elif endian == 'big':
        orig, data_flat = mock_sparse_data
        ds = raw_csr_generated_bigendian
        assert ds.dtype == np.dtype('>i4')
    else:
        raise ValueError()
    udf = SumUDF()
    roi = np.random.choice([True, False], data_flat.shape[0])
    res = lt_ctx.run_udf(udf=udf, dataset=ds, roi=roi)
    ref = for_backend(np.sum(data_flat[roi], axis=0), NUMPY)
    assert np.allclose(ref, res['intensity'].data.reshape((-1,)))


def test_raw_csr_ds_sumsig(raw_csr_generated, mock_sparse_data, lt_ctx):
    orig, data_flat = mock_sparse_data
    udf = SumSigUDF()
    res = lt_ctx.run_udf(udf=udf, dataset=raw_csr_generated)
    ref = for_backend(np.sum(data_flat, axis=(-1,)), NUMPY)
    assert np.allclose(ref, res['intensity'].data.reshape((-1,)))


def test_raw_csr_ds_sumsig_roi(raw_csr_generated, mock_sparse_data, lt_ctx):
    _, data_flat = mock_sparse_data
    udf = SumSigUDF()
    roi = np.random.choice([True, False], data_flat.shape[0])
    print("ROI size", np.count_nonzero(roi))
    # import pdb; pdb.set_trace()
    res = lt_ctx.run_udf(udf=udf, dataset=raw_csr_generated, roi=roi)
    ref = for_backend(np.sum(data_flat[roi], axis=(-1,)), NUMPY)
    assert np.allclose(ref, res['intensity'].raw_data.reshape((-1,)))


@pytest.fixture(scope="function")
def real_csr_data(lt_ctx):
    yield lt_ctx.load("raw_csr", path=RAW_CSR_TESTDATA_PATH)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CSR_TESTDATA, reason="need raw CSR testdata")
def test_sum_real_data(real_csr_data, local_cluster_ctx):
    udf = SumUDF()

    local_cluster_ctx.run_udf(udf=udf, dataset=real_csr_data)
    # ref = for_backend(np.sum(data_flat[roi], axis=0), NUMPY)
    # assert np.allclose(ref, res['intensity'].data.reshape((-1,)))


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CSR_TESTDATA, reason="need raw CSR testdata")
def test_sum_real_data_roi(real_csr_data, local_cluster_ctx):
    udf = SumUDF()

    roi = np.random.choice([True, False], real_csr_data.shape.nav)
    local_cluster_ctx.run_udf(udf=udf, dataset=real_csr_data, roi=roi)
    # ref = for_backend(np.sum(data_flat[roi], axis=0), NUMPY)
    # assert np.allclose(ref, res['intensity'].data.reshape((-1,)))


def test_detect_params(raw_csr_generated, default_raw, inline_executor):
    assert RawCSRDataSet.detect_params(raw_csr_generated._path, inline_executor)
    assert not RawCSRDataSet.detect_params(default_raw._path, inline_executor)


def test_diagnostics(raw_csr_generated):
    diags = raw_csr_generated.diagnostics
    assert {"name": "data dtype", "value": "float32"} in diags
    assert {"name": "indptr dtype", "value": "int32"} in diags
    assert {"name": "indices dtype", "value": "int32"} in diags


def test_exception_at_detect(tmpdir_factory, dask_executor):
    # setup: detect on valid UFT8, invalid TOML
    datadir = tmpdir_factory.mktemp('raw_csr_txt')
    fn = str(datadir / 'test.txt')
    with open(fn, "w") as f:
        f.write("stuff,in,here")

    # exceptions should be properly caught and should be pickleable:
    assert RawCSRDataSet.detect_params(fn, executor=dask_executor) is False


def test_sig_nav_shape(raw_csr_generated, lt_ctx):
    assert len(tuple(raw_csr_generated.shape.nav)) > 1
    assert len(tuple(raw_csr_generated.shape.sig)) > 1

    flat_nav = [prod(tuple(raw_csr_generated.shape.nav))]
    # throw in dict keys iterable for variety to test robustness for
    # heterogeneous types
    flat_sig = {
        prod(tuple(raw_csr_generated.shape.sig)): None
    }.keys()

    # Confirm types are incompatible
    with pytest.raises(TypeError):
        flat_nav + flat_sig

    ds = lt_ctx.load('auto', path=raw_csr_generated._path, nav_shape=flat_nav, sig_shape=flat_sig)
    udf = SumSigUDF()
    res = lt_ctx.run_udf(dataset=ds, udf=udf)
    ref = lt_ctx.run_udf(dataset=raw_csr_generated, udf=udf)

    assert np.allclose(
        ref['intensity'].data,
        res['intensity'].data.reshape(raw_csr_generated.shape.nav)
    )


@pytest.mark.slow
@pytest.mark.parametrize(
    'nav_shape', (
        None,
        (14, 14),
        (47, 13),
        "random1d",
        "random2d",
        "random3d",
    ),
)
@pytest.mark.parametrize(
    'sync_offset', (
        0, 1, -1, -10, 13, 13*15, -13*14,
        "random",
        -13*17+1,
        13*17-1,
    ),
)
@pytest.mark.parametrize(
    'sig_shape', (
        None, (24, 19), (24*19, )
    ),
)
@pytest.mark.parametrize(
    'use_roi', (False, True)
)
def test_reshape_sync_offset(
        raw_csr_generated, mock_sparse_data, lt_ctx, sync_offset, nav_shape, sig_shape, use_roi):
    if sync_offset == 'random':
        sync_offset = np.random.randint(low=-13*17+1, high=13*17)
    if nav_shape == 'random1d':
        nav_shape = (np.random.randint(low=1, high=13*18),)
    elif nav_shape == 'random2d':
        nav_shape = (
            np.random.randint(low=1, high=13*18),
            np.random.randint(low=1, high=13*18),
        )
    elif nav_shape == 'random3d':
        nav_shape = (
            np.random.randint(low=1, high=13*18),
            np.random.randint(low=1, high=13*18),
            np.random.randint(low=1, high=13*18),
        )

    orig, data_flat = mock_sparse_data
    data = data_flat.reshape(raw_csr_generated.shape)
    # Otherwise memory and raw_csr use different approach to determine shape
    # that yields different results
    if nav_shape is None and sig_shape is not None:
        mem_nav_shape = raw_csr_generated.shape.nav
    else:
        mem_nav_shape = nav_shape
    if sig_shape is None and nav_shape is not None:
        mem_sig_shape = raw_csr_generated.shape.sig
    else:
        mem_sig_shape = sig_shape
    ref_ds = lt_ctx.load(
        'memory',
        data=data,
        nav_shape=mem_nav_shape,
        sig_shape=mem_sig_shape,
        sync_offset=sync_offset,
    )

    if use_roi:
        roi = np.random.choice([True, False], size=ref_ds.shape.nav)
    else:
        roi = None

    print('nav_shape', nav_shape)
    print('sig_shape', sig_shape)
    print('sync_offset', sync_offset)
    print('roi', roi)

    ds = lt_ctx.load(
        'raw_csr', path=raw_csr_generated._path,
        sync_offset=sync_offset, nav_shape=nav_shape, sig_shape=sig_shape
    )

    assert tuple(ref_ds.shape.sig) == tuple(ds.shape.sig)
    assert tuple(ref_ds.shape.nav) == tuple(ds.shape.nav)

    masks = np.random.random(size=(3, *ref_ds.shape.sig))
    udf_masks = ApplyMasksUDF(mask_factories=lambda: masks)
    udf_std = StdDevUDF()

    ref_result = lt_ctx.run_udf(udf=(udf_masks, udf_std), dataset=ref_ds)
    result = lt_ctx.run_udf(udf=(udf_masks, udf_std), dataset=ds)

    r1 = ref_result[0]['intensity'].raw_data
    r2 = result[0]['intensity'].raw_data

    print(
        np.max((r1 - r2) / np.maximum(0.00001, (np.abs(r1) + np.abs(r2))))
    )

    assert_allclose(
        ref_result[0]['intensity'].raw_data,
        result[0]['intensity'].raw_data,
        rtol=1e-5,
        atol=1e-8,
    )
    assert_allclose(
        ref_result[1]['std'].raw_data,
        result[1]['std'].raw_data,
        rtol=1e-5,
        atol=1e-8,
    )
    assert_allclose(
        ref_result[1]['num_frames'].raw_data,
        result[1]['num_frames'].raw_data,
        rtol=1e-5,
        atol=1e-8,
    )


def test_uint64ptr(lt_ctx_fast, raw_csr_generated_uint64):
    prime_numba_cache(raw_csr_generated_uint64)


def test_large_file_detect(monkeypatch, default_raw, inline_executor_fast):
    # Use a mock load_toml function to check if we loaded a large file
    # could do this with a real mock object...
    load_called = False

    def mock_load_toml(*args, **kwargs):
        nonlocal load_called
        load_called = True

    monkeypatch.setattr(raw_csr, "load_toml", mock_load_toml)

    # default_raw is a 16 MB file
    filepath = default_raw._path
    detects = RawCSRDataSet.detect_params(filepath, inline_executor_fast)
    assert not detects
    assert not load_called


@pytest.mark.skipif(not HAVE_CSR_TESTDATA, reason="need raw CSR testdata")
def test_num_partitions(lt_ctx):
    ds = lt_ctx.load(
        "raw_csr",
        path=RAW_CSR_TESTDATA_PATH,
        num_partitions=129,
    )
    assert len(list(ds.get_partitions())) == 129
