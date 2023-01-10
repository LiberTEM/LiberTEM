import os

from scipy.sparse import csr_matrix
import numpy as np
import pytest

from libertem.io.dataset.raw_csr import (
    read_tiles_straight, read_tiles_with_roi, CSRTriple, RawCSRDataSet
)
from sparseconverter import NUMPY, for_backend, SCIPY_CSR
from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape, Slice
from libertem.udf.sum import SumUDF
from libertem.udf.sumsigudf import SumSigUDF
from libertem.common.math import prod

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
    ):
        res += np.sum(tile.data, axis=0)

        total_size += tile.data.shape[0]
        assert tile.shape[0] <= tuple(tileshape)[0]
        assert isinstance(tile.data, csr_matrix)
    assert total_size == np.count_nonzero(roi_slice)

    assert np.allclose(ref, res)


def test_raw_csr_ds_sum(raw_csr_generated, mock_sparse_data, lt_ctx):
    orig, data_flat = mock_sparse_data
    udf = SumUDF()
    res = lt_ctx.run_udf(udf=udf, dataset=raw_csr_generated)
    ref = for_backend(np.sum(data_flat, axis=0), NUMPY)
    assert np.allclose(ref, res['intensity'].data.reshape((-1,)))


def test_raw_csr_ds_sum_roi(raw_csr_generated, mock_sparse_data, lt_ctx):
    _, data_flat = mock_sparse_data
    udf = SumUDF()
    roi = np.random.choice([True, False], data_flat.shape[0])
    res = lt_ctx.run_udf(udf=udf, dataset=raw_csr_generated, roi=roi)
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


@pytest.mark.skipif(not HAVE_CSR_TESTDATA, reason="need raw CSR testdata")
def test_sum_real_data(real_csr_data, lt_ctx):
    udf = SumUDF()

    roi = np.random.choice([True, False], real_csr_data.shape.nav)
    lt_ctx.run_udf(udf=udf, dataset=real_csr_data, roi=roi)
    # ref = for_backend(np.sum(data_flat[roi], axis=0), NUMPY)
    # assert np.allclose(ref, res['intensity'].data.reshape((-1,)))


def test_detect_params(raw_csr_generated, default_raw, inline_executor):
    assert RawCSRDataSet.detect_params(raw_csr_generated._path, inline_executor)
    assert not RawCSRDataSet.detect_params(default_raw._path, inline_executor)


def test_diagnostics(raw_csr_generated):
    raw_csr_generated.diagnostics


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
