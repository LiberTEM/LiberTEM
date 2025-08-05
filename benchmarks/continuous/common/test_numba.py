import scipy.sparse
import sparse
import pytest
import numpy as np

from libertem.common.numba import rmatmul

# Adjust to scale benchmark:
N = 3*1024
M = N
L = M // 8
K = L // 8


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_rmatmul_csr(benchmark):
    data = np.zeros((2*N, N), dtype=np.float32)
    masks = scipy.sparse.csr_matrix(
        ([1.]*L, (range(0, M, 8), [0, 1, 2, 3, 4, 5, 6, 7]*K)),
        shape=(N, 8),
        dtype=np.float32
    )
    benchmark(rmatmul, data, masks)


# @pytest.mark.slow
@pytest.mark.benchmark(
    group="rmatmul",
)
def test_scipysparse_csr_right(benchmark):
    data = np.zeros((2*N, N), dtype=np.float32)
    masks = scipy.sparse.csr_matrix(
        ([1.]*L, (range(0, M, 8), [0, 1, 2, 3, 4, 5, 6, 7]*K)),
        shape=(N, 8),
        dtype=np.float32
    )

    def doit(data, masks):
        return data @ masks

    benchmark(doit, data, masks)


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_scipysparse_csr_left(benchmark):
    data = np.zeros((N, 2*N), dtype=np.float32)
    masks = scipy.sparse.csr_matrix(
        ([1.]*L, ([0, 1, 2, 3, 4, 5, 6, 7]*K, range(0, M, 8))),
        shape=(8, N),
        dtype=np.float32
    )

    def doit(data, masks):
        return masks @ data

    benchmark(doit, data, masks)


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_rmatmul_csc(benchmark):
    data = np.zeros((2*N, N), dtype=np.float32)
    masks = scipy.sparse.csc_matrix(
        ([1.]*L, (range(0, M, 8), [0, 1, 2, 3, 4, 5, 6, 7]*K)),
        shape=(N, 8),
        dtype=np.float32
    )
    benchmark(rmatmul, data, masks)


# @pytest.mark.slow
@pytest.mark.benchmark(
    group="rmatmul",
)
def test_scipysparse_csc_right(benchmark):
    data = np.zeros((2*N, N), dtype=np.float32)
    masks = scipy.sparse.csc_matrix(
        ([1.]*L, (range(0, M, 8), [0, 1, 2, 3, 4, 5, 6, 7]*K)),
        shape=(N, 8),
        dtype=np.float32
    )

    def doit(data, masks):
        return data @ masks

    benchmark(doit, data, masks)


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_scipysparse_csc_left(benchmark):
    data = np.zeros((N, 2*N), dtype=np.float32)
    masks = scipy.sparse.csc_matrix(
        ([1.]*L, ([0, 1, 2, 3, 4, 5, 6, 7]*K, range(0, M, 8))),
        shape=(8, N),
        dtype=np.float32
    )

    def doit(data, masks):
        return masks @ data

    benchmark(doit, data, masks)


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_sparse_coo_right(benchmark):
    data = np.zeros((2*N, N), dtype=np.float32)
    masks = sparse.COO(scipy.sparse.csr_matrix(
        ([1.]*L, (range(0, M, 8), [0, 1, 2, 3, 4, 5, 6, 7]*K)),
        shape=(N, 8),
        dtype=np.float32
    ))

    def doit(data, masks):
        return data @ masks

    benchmark(doit, data, masks)


@pytest.mark.benchmark(
    group="rmatmul",
)
def test_sparse_coo_left(benchmark):
    data = np.zeros((N, 2*N), dtype=np.float32)
    masks = sparse.COO(scipy.sparse.csr_matrix(
        ([1.]*L, ([0, 1, 2, 3, 4, 5, 6, 7]*K, range(0, M, 8))),
        shape=(8, N),
        dtype=np.float32
    ))

    def doit(data, masks):
        return masks @ data

    benchmark(doit, data, masks)
