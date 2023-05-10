import os
import glob
import pickle
import shutil
import importlib

import scipy.sparse
import pytest
import numpy as np
import numba

from libertem.api import Context
from libertem.common.numba import rmatmul, numba_dtypes
from libertem.common.numba.cache import _cached_njit_reg
from libertem.web.dataset import prime_numba_cache


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


def test_numba_cache(tmpdir_factory):
    src_dir = tmpdir_factory.mktemp('numba_cache')
    src_py_file = os.path.join(os.path.dirname(__file__), 'numba_cache.py')
    shutil.copy(src_py_file, src_dir)

    dst_py_file = os.path.join(src_dir, 'numba_cache.py')
    spec = importlib.util.spec_from_file_location('<dynamic>', dst_py_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    pycache_dir = os.path.join(src_dir, '__pycache__')

    nbis = glob.glob(pycache_dir + '/*.nbi')
    nbcs = glob.glob(pycache_dir + '/*.nbc')

    assert len(_cached_njit_reg) >= 1
    assert len(nbis) == 1
    assert len(nbcs) == 2

    with open(nbis[0], 'rb') as f:
        pickle.load(f)  # load and discard version
        data = f.read()
        cache_contents = pickle.loads(data)

    assert len(cache_contents[1].keys()) == 2


@pytest.mark.slow
def test_numba_prime(default_raw):
    prime_numba_cache(default_raw)


def test_numba_prime_hdf5_1(hdf5_ds_1):
    prime_numba_cache(hdf5_ds_1)


def test_numba_prime_hdf5_2(hdf5_ds_large_sig):
    prime_numba_cache(hdf5_ds_large_sig)


@numba.njit
def numba_sum(arr):
    return arr.sum()


@pytest.mark.parametrize(
    'dtype', numba_dtypes
)
def test_numba_dtype(dtype):
    arr = np.ones(42, dtype=dtype)
    res = numba_sum(arr)
    assert arr.dtype in numba_dtypes
    assert res == len(arr)


@numba.njit
def _issue_1432():
    np.array([1 for _ in range(1)])


@pytest.mark.slow
def test_issue_1432():
    from libertem.executor.dask import cluster_spec, DaskJobExecutor
    spec = cluster_spec(cpus=1, cudas=[], has_cupy=False, num_service=0)
    executor = DaskJobExecutor.make_local(spec=spec)
    with Context(executor=executor) as ctx:
        print(ctx.executor.run_function(_issue_1432))
