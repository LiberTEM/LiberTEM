from unittest import mock

from scipy.sparse import csr_matrix
import numpy as np
import pytest

from libertem.io.dataset.raw_csr import (
    CSRDescriptor, read_tiles_straight, read_tiles_with_roi, CSRTriple,
    RawCSRDataSet,
)
from libertem.common.array_backends import NUMPY, for_backend, SCIPY_CSR
from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape, Slice
from libertem.udf.sum import SumUDF

from utils import _mk_random


def test_get_tiles_straight():
    data = _mk_random((13, 17, 24, 19), array_backend=NUMPY)
    data_flat = data.reshape((13*17, 24*19))

    sig_shape = (24, 19)
    tileshape = Shape((11, ) + sig_shape, sig_dims=2)
    dataset_shape = Shape(data.shape, sig_dims=2)

    orig = for_backend(data_flat, SCIPY_CSR)

    orig_triple = CSRTriple(indptr=orig.indptr, coords=orig.indices, values=orig.data)
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=dataset_shape,
        intent='tile'
    )
    partition_slice = Slice(origin=(3, 0, 0), shape=Shape((51, ) + sig_shape, sig_dims=2))

    ref = for_backend(np.sum(data_flat[3:3+51], axis=0), NUMPY)

    res = np.zeros((1, data_flat.shape[1]), dtype=data.dtype)

    for tile in read_tiles_straight(
            triple=orig_triple, partition_slice=partition_slice, tiling_scheme=tiling_scheme
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

    orig_triple = CSRTriple(indptr=orig.indptr, coords=orig.indices, values=orig.data)
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=dataset_shape,
        intent='tile'
    )
    partition_slice = Slice(origin=(1, 0), shape=Shape((3, ) + sig_shape, sig_dims=1))

    ref = for_backend(np.sum(data_flat[1:], axis=0), NUMPY)

    res = np.zeros((1, data_flat.shape[1]), dtype=data.dtype)

    for tile in read_tiles_straight(
            triple=orig_triple, partition_slice=partition_slice, tiling_scheme=tiling_scheme
    ):
        res += np.sum(tile.data, axis=0)
        assert tile.shape[0] <= tuple(tileshape)[0]
        assert isinstance(tile.data, csr_matrix)

    assert np.allclose(ref, res)


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

    orig_triple = CSRTriple(indptr=orig.indptr, coords=orig.indices, values=orig.data)
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
            triple=orig_triple, partition_slice=partition_slice, tiling_scheme=tiling_scheme,
            roi=roi
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

    orig_triple = CSRTriple(indptr=orig.indptr, coords=orig.indices, values=orig.data)
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

    for tile in read_tiles_with_roi(
            triple=orig_triple,
            partition_slice=partition_slice,
            tiling_scheme=tiling_scheme,
            roi=roi
    ):
        res += np.sum(tile.data, axis=0)
        assert tile.shape[0] <= tuple(tileshape)[0]
        assert isinstance(tile.data, csr_matrix)

    assert np.allclose(ref, res)


@pytest.fixture(scope="module")
def mock_sparse_data():
    data = _mk_random((13, 17, 24, 19), array_backend=NUMPY)
    data_flat = data.reshape((13*17, 24*19))
    orig = for_backend(data_flat, SCIPY_CSR)
    return orig, data_flat


@pytest.fixture
def mock_csr_triple(mock_sparse_data):
    orig, _ = mock_sparse_data
    triple = CSRTriple(
        indptr=orig.indptr,
        coords=orig.indices,
        values=orig.data
    )

    with mock.patch('libertem.io.dataset.raw_csr.get_triple', side_effect=lambda x: triple):
        yield triple


def test_raw_csr_ds_sum(mock_csr_triple: CSRTriple, mock_sparse_data, lt_ctx):
    orig, data_flat = mock_sparse_data
    desc = CSRDescriptor(
        indptr_file="",
        indptr_dtype=mock_csr_triple.indptr.dtype,
        coords_file="",
        coords_dtype=mock_csr_triple.coords.dtype,
        values_file="",
        values_dtype=mock_csr_triple.values.dtype,
    )
    ds = RawCSRDataSet(descriptor=desc, nav_shape=(13, 17), sig_shape=(24, 19), io_backend=None)
    ds = ds.initialize(executor=lt_ctx.executor)
    udf = SumUDF()

    res = lt_ctx.run_udf(udf=udf, dataset=ds)
    ref = for_backend(np.sum(data_flat, axis=0), NUMPY)
    assert np.allclose(ref, res['intensity'].data.reshape((-1,)))


def test_raw_csr_ds_sum_roi(mock_csr_triple: CSRTriple, mock_sparse_data, lt_ctx):
    orig, data_flat = mock_sparse_data
    desc = CSRDescriptor(
        indptr_file="",
        indptr_dtype=mock_csr_triple.indptr.dtype,
        coords_file="",
        coords_dtype=mock_csr_triple.coords.dtype,
        values_file="",
        values_dtype=mock_csr_triple.values.dtype,
    )
    ds = RawCSRDataSet(descriptor=desc, nav_shape=(13, 17), sig_shape=(24, 19), io_backend=None)
    ds = ds.initialize(executor=lt_ctx.executor)
    udf = SumUDF()

    roi = np.random.choice([True, False], data_flat.shape[0])
    res = lt_ctx.run_udf(udf=udf, dataset=ds, roi=roi)
    ref = for_backend(np.sum(data_flat[roi], axis=0), NUMPY)
    assert np.allclose(ref, res['intensity'].data.reshape((-1,)))
