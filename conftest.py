import os
import importlib.util

import numpy as np
import pytest
import h5py

from libertem.io.dataset.hdf5 import H5DataSet
from libertem.io.dataset.raw import RawFileDataSet
from libertem.io.dataset.memory import MemoryDataSet


# A bit of gymnastics to import the test utilities since this
# conftest.py file is shared between the doctests and unit tests
basedir = os.path.dirname(__file__)
location = os.path.join(basedir, "tests/utils.py")
spec = importlib.util.spec_from_file_location("utils", location)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


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
def random_hdf5(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/hdf5-test-random.h5'
    try:
        with h5py.File(filename, 'r') as f:
            yield f
    except OSError:
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", data=utils._mk_random(size=(5, 5, 16, 16), dtype="float32"))
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
        path=hdf5.filename, ds_path="data", tileshape=(1, 5, 16, 16)
    )
    ds = ds.initialize()
    return ds


@pytest.fixture
def hdf5_ds_2(random_hdf5):
    ds = H5DataSet(
        path=random_hdf5.filename, ds_path="data", tileshape=(1, 5, 16, 16)
    )
    ds = ds.initialize()
    return ds


@pytest.fixture
def ds_complex():
    data = np.random.choice(
        a=[0, 1, 0+1j, 0-1j, 1+1j, 1-1j], size=(16, 16, 16, 16)
    ).astype('complex64')
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2,
    )
    return dataset


@pytest.fixture(scope='session')
def default_raw(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-default'
    data = utils._mk_random(size=(16, 16, 128, 128), dtype='float32')
    data.tofile(str(filename))
    del data
    ds = RawFileDataSet(
        path=str(filename),
        scan_size=(16, 16),
        dtype="float32",
        detector_size=(128, 128),
    )
    ds.set_num_cores(2)
    ds = ds.initialize()
    yield ds


@pytest.fixture(scope='session')
def uint16_raw(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-default'
    data = utils._mk_random(size=(16, 16, 128, 128), dtype='uint16')
    data.tofile(str(filename))
    del data
    ds = RawFileDataSet(
        path=str(filename),
        scan_size=(16, 16),
        dtype="uint16",
        detector_size=(128, 128),
    )
    ds = ds.initialize()
    yield ds
