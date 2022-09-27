from scipy.sparse import csr_matrix
import numpy as np

from libertem.io.dataset.raw_csr import read_tiles_straight, CSRTriple
from libertem.common.array_backends import NUMPY, for_backend, SCIPY_CSR
from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape, Slice

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
