import scipy.sparse
import pytest
import numpy as np

from libertem.common.numba import rmatmul


@pytest.mark.with_numba
def test_rmatmul_csr():
    le = np.random.random((2, 3))
    ri = scipy.sparse.csr_matrix(np.random.random((3, 2)))
    assert np.allclose(rmatmul(le, ri), le @ ri)


@pytest.mark.with_numba
def test_rmatmul_csc():
    le = np.random.random((2, 3))
    ri = scipy.sparse.csr_matrix(np.random.random((3, 2)))
    assert np.allclose(rmatmul(le, ri), le @ ri)


def test_rmatmul_1():
    le = np.zeros((1, 2, 3))
    ri = scipy.sparse.csr_matrix(np.zeros((5, 6)))
    # 3D shape left
    with pytest.raises(ValueError):
        rmatmul(le, ri)


def test_rmatmul_2():
    le = np.zeros((2, 3))
    ri = np.zeros((4, 5, 6))
    # 3D shape right
    with pytest.raises(ValueError, ):
        rmatmul(le, ri)


def test_rmatmul_3():
    le = np.zeros((2, 3))
    ri = scipy.sparse.csr_matrix(np.zeros((5, 6)))
    # Shape mismatch
    with pytest.raises(ValueError):
        rmatmul(le, ri)


def test_rmatmul_4():
    le = np.zeros((2, 3))
    ri = np.zeros((3, 2))
    # Not a csc or csr matrix
    with pytest.raises(ValueError):
        rmatmul(le, ri)
