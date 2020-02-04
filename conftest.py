import os
import importlib.util
import platform
import shutil

import numpy as np
import pytest
import h5py

import libertem.api as lt
from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.hdf5 import H5DataSet
from libertem.io.dataset.raw import RawFileDataSet
from libertem.io.dataset.memory import MemoryDataSet
from libertem.executor.dask import DaskJobExecutor


# A bit of gymnastics to import the test utilities since this
# conftest.py file is shared between the doctests and unit tests
# and this file is outside the package
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
def hdf5_3D(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/hdf5-test-3D.h5'
    try:
        with h5py.File(filename, 'r') as f:
            yield f
    except OSError:
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", data=np.ones((17, 16, 16)))
        with h5py.File(filename, 'r') as f:
            yield f


@pytest.fixture(scope='session')
def hdf5_5D(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/hdf5-test-5D.h5'
    try:
        with h5py.File(filename, 'r') as f:
            yield f
    except OSError:
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", data=np.ones((3, 5, 9, 16, 16)))
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
    ds = ds.initialize(InlineJobExecutor())
    return ds


@pytest.fixture
def hdf5_ds_2(random_hdf5):
    ds = H5DataSet(
        path=random_hdf5.filename, ds_path="data", tileshape=(1, 5, 16, 16)
    )
    ds = ds.initialize(InlineJobExecutor())
    return ds


@pytest.fixture
def hdf5_ds_3D(hdf5_3D):
    ds = H5DataSet(
        path=hdf5_3D.filename, ds_path="data", tileshape=(1, 16, 16)
    )
    ds = ds.initialize(InlineJobExecutor())
    return ds


@pytest.fixture
def hdf5_ds_5D(hdf5_5D):
    ds = H5DataSet(
        path=hdf5_5D.filename, ds_path="data", tileshape=(1, 1, 1, 16, 16)
    )
    ds = ds.initialize(InlineJobExecutor())
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
    ds = ds.initialize(InlineJobExecutor())
    yield ds


@pytest.fixture
def raw_on_workers(dist_ctx, tmpdir_factory):
    """
    copy raw dataset to each worker
    """

    datadir = tmpdir_factory.mktemp('data')
    filename = str(datadir + '/raw-test-on-workers')

    data = utils._mk_random(size=(16, 16, 128, 128), dtype='float32')

    tmpdirpath = os.path.dirname(filename)

    def _make_example_raw():
        # workers don't automatically have the pytest tmp directory, create it:
        if not os.path.exists(tmpdirpath):
            os.makedirs(tmpdirpath)
        print("creating %s" % filename)
        data.tofile(filename)
        print("created %s" % filename)
        return tmpdirpath, os.listdir(tmpdirpath)

    import cloudpickle
    import pickle
    dumped = cloudpickle.dumps(_make_example_raw)

    pickle.loads(dumped)

    print("raw_on_workers _make_example_raw: %s" %
          (dist_ctx.executor.run_each_host(_make_example_raw),))

    ds = dist_ctx.load("raw",
                       path=str(filename),
                       scan_size=(16, 16),
                       detector_size=(128, 128),
                       dtype="float32")

    yield ds

    def _cleanup():
        # FIXME: this may litter /tmp/ with empty directories, as we only remove our own
        # tmpdirpath, but as we run these tests in docker containers, they are eventually
        # cleaned up anyways:
        files = os.listdir(tmpdirpath)
        shutil.rmtree(tmpdirpath, ignore_errors=True)
        print("removed %s" % tmpdirpath)
        return tmpdirpath, files

    print("raw_on_workers cleanup: %s" % (dist_ctx.executor.run_each_host(_cleanup),))


@pytest.fixture(scope='session')
def large_raw(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-large-sparse'
    shape = (100, 100, 1216, 1216)
    dtype = np.uint16
    size = np.prod(np.int64(shape)) * np.dtype(dtype).itemsize
    if platform.system() == "Windows":
        os.system('FSUtil File CreateNew "%s" 0x%X' % (filename, size))
        os.system('FSUtil Sparse SetFlag "%s"' % filename)
        os.system('FSUtil Sparse SetRange "%s" 0 0x%X' % (filename, size))
    else:
        with open(filename, 'wb') as f:
            f.truncate(size)
        stat = os.stat(filename)
        assert stat.st_blocks == 0
    ds = RawFileDataSet(
        path=str(filename),
        scan_size=shape[:2],
        dtype=dtype,
        detector_size=shape[2:],
    )
    ds.set_num_cores(2)
    ds = ds.initialize(InlineJobExecutor())
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
    ds = ds.initialize(InlineJobExecutor())
    yield ds


@pytest.fixture
def dist_ctx():
    """
    This Context needs to have an external dask cluster running, with the following
    assumptions:

     - two workers: hostnames worker-1 and worker-2
     - one scheduler node
     - data availability TBD
     - the address of the dask scheduler is passed in as DASK_SCHEDULER_ADDRESS
    """
    scheduler_addr = os.environ['DASK_SCHEDULER_ADDRESS']
    executor = DaskJobExecutor.connect(scheduler_addr)
    with lt.Context(executor=executor) as ctx:
        yield ctx


@pytest.fixture(autouse=True)
def auto_ctx(doctest_namespace):
    ctx = lt.Context(executor=InlineJobExecutor())
    doctest_namespace["ctx"] = ctx


@pytest.fixture(autouse=True)
def auto_ds(doctest_namespace):
    dataset = MemoryDataSet(datashape=[16, 16, 16, 16])
    doctest_namespace["dataset"] = dataset


@pytest.fixture(autouse=True)
def auto_libs(doctest_namespace):
    doctest_namespace["np"] = np


@pytest.fixture(autouse=True)
def auto_libertem(doctest_namespace):
    import libertem
    import libertem.utils
    import libertem.utils.generate
    import libertem.udf.blobfinder
    import libertem.masks
    import libertem.api
    doctest_namespace["libertem"] = libertem
    doctest_namespace["libertem.utils"] = libertem.utils
    doctest_namespace["libertem.utils.generate"] = libertem.utils.generate
    doctest_namespace["libertem.udf.blobfinder"] = libertem.udf.blobfinder
    doctest_namespace["libertem.masks"] = libertem.masks
    doctest_namespace["libertem.api"] = libertem.api


@pytest.fixture(autouse=True)
def auto_files(doctest_namespace, hdf5, default_raw):
    doctest_namespace["path_to_hdf5"] = hdf5.filename
    doctest_namespace["path_to_raw"] = default_raw._path
