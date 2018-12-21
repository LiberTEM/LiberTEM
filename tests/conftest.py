import pytest
import h5py
import numpy as np

from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.hdf5 import H5DataSet
from libertem import api as lt

from utils import MemoryDataSet


@pytest.fixture
def inline_executor():
    return InlineJobExecutor()


@pytest.fixture
def lt_ctx(inline_executor):
    return lt.Context(executor=inline_executor)


@pytest.fixture(scope='session')
def hdf5(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/hdf5-test.h5'
    try:
        with h5py.File(filename, 'r') as f:
            yield f
    except OSError:
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", data=np.ones((5, 5, 16, 16)))
        with h5py.File(filename, 'r') as f:
            yield f


@pytest.fixture(scope='session')
def empty_hdf5(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/hdf5-empty.h5'
    try:
        with h5py.File(filename, 'r') as f:
            yield f
    except OSError:
        with h5py.File(filename, "w") as f:
            pass
        with h5py.File(filename, 'r') as f:
            yield f


@pytest.fixture
def hdf5_ds_1(hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data", tileshape=(1, 5, 16, 16), target_size=512*1024*1024
    )
    ds.initialize()
    return ds


@pytest.fixture
def ds_complex():
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype('complex64')
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 1, 16, 16),
        partition_shape=(16, 16, 16, 16)
    )
    return dataset
