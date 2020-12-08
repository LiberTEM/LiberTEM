import scipy.sparse
import sparse
import pytest
import numpy as np

from libertem.common.numba import rmatmul


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_rmatmul_csr(benchmark):
    data = np.zeros((2*16384, 16384), dtype=np.float32)
    masks = scipy.sparse.csr_matrix(
        ([1.]*1000, (range(0, 8000, 8), [0, 1, 2, 3, 4, 5, 6, 7]*125)),
        shape=(16384, 8),
        dtype=np.float32
    )
    benchmark(rmatmul, data, masks)


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_scipysparse_csr_right(benchmark):
    data = np.zeros((2*16384, 16384), dtype=np.float32)
    masks = scipy.sparse.csr_matrix(
        ([1.]*1000, (range(0, 8000, 8), [0, 1, 2, 3, 4, 5, 6, 7]*125)),
        shape=(16384, 8),
        dtype=np.float32
    )

    def doit(data, masks):
        return data @ masks

    benchmark(doit, data, masks)


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_scipysparse_csr_left(benchmark):
    data = np.zeros((16384, 2*16384), dtype=np.float32)
    masks = scipy.sparse.csr_matrix(
        ([1.]*1000, ([0, 1, 2, 3, 4, 5, 6, 7]*125, range(0, 8000, 8))),
        shape=(8, 16384),
        dtype=np.float32
    )

    def doit(data, masks):
        return masks @ data

    benchmark(doit, data, masks)


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_rmatmul_csc(benchmark):
    data = np.zeros((2*16384, 16384), dtype=np.float32)
    masks = scipy.sparse.csc_matrix(
        ([1.]*1000, (range(0, 8000, 8), [0, 1, 2, 3, 4, 5, 6, 7]*125)),
        shape=(16384, 8),
        dtype=np.float32
    )
    benchmark(rmatmul, data, masks)


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_scipysparse_csc_right(benchmark):
    data = np.zeros((2*16384, 16384), dtype=np.float32)
    masks = scipy.sparse.csc_matrix(
        ([1.]*1000, (range(0, 8000, 8), [0, 1, 2, 3, 4, 5, 6, 7]*125)),
        shape=(16384, 8),
        dtype=np.float32
    )

    def doit(data, masks):
        return data @ masks

    benchmark(doit, data, masks)


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_scipysparse_csc_left(benchmark):
    data = np.zeros((16384, 2*16384), dtype=np.float32)
    masks = scipy.sparse.csc_matrix(
        ([1.]*1000, ([0, 1, 2, 3, 4, 5, 6, 7]*125, range(0, 8000, 8))),
        shape=(8, 16384),
        dtype=np.float32
    )

    def doit(data, masks):
        return masks @ data

    benchmark(doit, data, masks)


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_sparse_coo_right(benchmark):
    data = np.zeros((2*16384, 16384), dtype=np.float32)
    masks = sparse.COO(scipy.sparse.csr_matrix(
        ([1.]*1000, (range(0, 8000, 8), [0, 1, 2, 3, 4, 5, 6, 7]*125)),
        shape=(16384, 8),
        dtype=np.float32
    ))

    def doit(data, masks):
        return data @ masks

    benchmark(doit, data, masks)


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_sparse_coo_left(benchmark):
    data = np.zeros((16384, 2*16384), dtype=np.float32)
    masks = sparse.COO(scipy.sparse.csr_matrix(
        ([1.]*1000, ([0, 1, 2, 3, 4, 5, 6, 7]*125, range(0, 8000, 8))),
        shape=(8, 16384),
        dtype=np.float32
    ))

    def doit(data, masks):
        return masks @ data

    benchmark(doit, data, masks)
