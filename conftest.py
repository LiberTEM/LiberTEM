import asyncio
import sys
import os
import importlib.util
import platform
import threading
from functools import partial
import warnings
import contextlib
import socket
import logging
import textwrap

from importlib.metadata import distributions
import numpy as np
import pytest
import h5py
import aiohttp
from dask import distributed as dd
import dask
from scipy.sparse import csr_matrix
from sparseconverter import NUMPY, SCIPY_CSR, for_backend

from distributed.scheduler import Scheduler
import tornado.httpserver

import libertem.api as lt
from libertem.executor.inline import InlineJobExecutor
from libertem.executor.delayed import DelayedJobExecutor
from libertem.io.dataset import get_dataset_cls
from libertem.io.dataset.hdf5 import H5DataSet
from libertem.io.dataset.raw import RawFileDataSet
from libertem.io.dataset.memory import MemoryDataSet
from libertem.io.dataset.base import BufferedBackend, MMapBackend, DirectBackend
from libertem.executor.dask import DaskJobExecutor, cluster_spec
from libertem.executor.concurrent import ConcurrentJobExecutor
from libertem.common.threading import set_num_threads_env
from libertem.viz.base import Dummy2DPlot

from libertem.utils.devices import detect

from libertem.web.server import make_app, EventRegistry
from libertem.web.state import SharedState, ExecutorState
from libertem.web.event_bus import EventBus
from libertem.executor.base import AsyncAdapter
from libertem.common.async_utils import sync_to_async
from libertem.common.async_utils import adjust_event_loop_policy
from libertem.common.tracing import maybe_setup_tracing

maybe_setup_tracing("pytest")


# A bit of gymnastics to import the test utilities since this
# conftest.py file is shared between the doctests and unit tests
# and this file is outside the package
basedir = os.path.dirname(__file__)
location = os.path.join(basedir, "tests/utils.py")
spec = importlib.util.spec_from_file_location("utils", location)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)

# Mitigation for https://github.com/dask/distributed/issues/6776
dask.config.set({"distributed.worker.profile.enabled": False})


def get_or_create_hdf5(tmpdir_factory, filename, *args, **kwargs):
    datadir = tmpdir_factory.mktemp('data')
    filename = os.path.join(datadir, filename)
    try:
        with h5py.File(filename, 'r') as f:
            yield f
    except OSError:
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", *args, **kwargs)
        with h5py.File(filename, 'r') as f:
            yield f


@pytest.fixture(scope='session')
def hdf5(tmpdir_factory):
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test.h5", data=np.ones((5, 5, 16, 16)))


@pytest.fixture(scope='session')
def hdf5_2d(tmpdir_factory):
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-2d.h5", data=np.ones((16, 16)))


@pytest.fixture(scope='session')
def hdf5_3d(tmpdir_factory):
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-3d.h5", data=np.ones((17, 16, 16)))


@pytest.fixture(scope='session')
def hdf5_5d(tmpdir_factory):
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-5d.h5",
                                  data=np.ones((3, 5, 9, 16, 16)))


@pytest.fixture(scope='session')
def random_hdf5_large_sig(tmpdir_factory):
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-random.h5",
                                  data=np.random.randn(16, 16, 512, 512))


@pytest.fixture(scope='session')
def random_hdf5(tmpdir_factory):
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-random.h5",
                                  data=np.random.randn(5, 5, 16, 16))


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


@pytest.fixture(scope='session')
def hdf5_4d_data():
    data = np.random.random((2, 10, 26, 26)).astype("float32")
    yield data


@pytest.fixture(scope='session')
def hdf5_same_data_3d(tmpdir_factory, hdf5_4d_data):
    data = hdf5_4d_data.reshape((20, 26, 26))
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-reshape-3d.h5", data=data)


@pytest.fixture(scope='session')
def hdf5_same_data_4d(tmpdir_factory, hdf5_4d_data):
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-reshape-4d.h5", data=hdf5_4d_data)


@pytest.fixture(scope='session')
def hdf5_same_data_5d(tmpdir_factory, hdf5_4d_data):
    data = hdf5_4d_data.reshape((2, 2, 5, 26, 26))
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-reshape-5d.h5", data=data)


@pytest.fixture(scope='session')
def hdf5_same_data_1d_sig(tmpdir_factory, hdf5_4d_data):
    data = hdf5_4d_data.reshape((2, 10, 676))
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-reshape-1d-sig.h5", data=data)


@pytest.fixture(scope='session')
def raw_same_dataset_4d(tmpdir_factory, hdf5_4d_data):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-same-data-4d'
    hdf5_4d_data.tofile(str(filename))
    ds = RawFileDataSet(
        path=str(filename),
        nav_shape=(2, 10),
        dtype="float32",
        sig_shape=(26, 26),
    )
    ds.set_num_cores(4)
    ds = ds.initialize(InlineJobExecutor())
    yield ds


@pytest.fixture
def hdf5_ds_1(hdf5):
    ds = H5DataSet(
        path=hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(InlineJobExecutor())
    return ds


@pytest.fixture
def hdf5_ds_2(random_hdf5):
    ds = H5DataSet(
        path=random_hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(InlineJobExecutor())
    return ds


@pytest.fixture
def hdf5_ds_large_sig(random_hdf5):
    ds = H5DataSet(
        path=random_hdf5.filename, ds_path="data",
    )
    ds = ds.initialize(InlineJobExecutor())
    return ds


@pytest.fixture
def hdf5_ds_3d(hdf5_3d):
    ds = H5DataSet(
        path=hdf5_3d.filename, ds_path="data",
    )
    ds = ds.initialize(InlineJobExecutor())
    return ds


@pytest.fixture
def hdf5_ds_2d(hdf5_2d):
    ds = H5DataSet(
        path=hdf5_2d.filename, ds_path="data",
    )
    ds = ds.initialize(InlineJobExecutor())
    return ds


@pytest.fixture
def hdf5_ds_5d(hdf5_5d):
    ds = H5DataSet(
        path=hdf5_5d.filename, ds_path="data",
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
def default_raw_data():
    return utils._mk_random(size=(16, 16, 128, 128), dtype='float32')


@pytest.fixture(scope='session')
def default_raw_file(tmpdir_factory, default_raw_data):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-default'
    default_raw_data.tofile(str(filename))
    del default_raw_data
    return filename


@pytest.fixture(scope='session')
def default_raw(default_raw_file):
    lt_ctx = lt.Context(executor=InlineJobExecutor())
    ds = lt_ctx.load(
        "raw",
        path=str(default_raw_file),
        dtype="float32",
        nav_shape=(16, 16),
        sig_shape=(128, 128),
        io_backend=MMapBackend(),
    )
    ds.set_num_cores(2)
    try:
        yield ds
    finally:
        lt_ctx.close()


@pytest.fixture(scope='session')
def prime_raw_data():
    return utils._mk_random(size=(13, 17, 19, 23), dtype='float32')


@pytest.fixture(scope='session')
def prime_raw(tmpdir_factory, prime_raw_data):
    lt_ctx = lt.Context(executor=InlineJobExecutor())
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-prime'
    prime_raw_data.tofile(str(filename))
    del prime_raw_data
    ds = lt_ctx.load(
        "raw",
        path=str(filename),
        dtype="float32",
        nav_shape=(13, 17),
        sig_shape=(19, 23),
        io_backend=MMapBackend(),
    )
    ds.set_num_cores(2)
    try:
        yield ds
    finally:
        lt_ctx.close()


@pytest.fixture(scope='session')
def default_raw_asymm(tmpdir_factory, default_raw_data):
    lt_ctx = lt.Context(executor=InlineJobExecutor())
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-default'
    default_raw_data.tofile(str(filename))
    del default_raw_data
    ds = lt_ctx.load(
        "raw",
        path=str(filename),
        dtype="float32",
        nav_shape=(14, 17),
        sig_shape=(128, 128),
        io_backend=MMapBackend(),
    )
    ds.set_num_cores(2)
    yield ds


@pytest.fixture(scope='session')
def buffered_raw(tmpdir_factory, default_raw_data):
    lt_ctx = lt.Context(executor=InlineJobExecutor())
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-buffered'
    default_raw_data.tofile(str(filename))
    del default_raw_data

    ds = lt_ctx.load(
        "raw",
        path=str(filename),
        dtype="float32",
        nav_shape=(16, 16),
        sig_shape=(128, 128),
        io_backend=BufferedBackend(),
    )
    yield ds


@pytest.fixture(scope='session')
def direct_raw(tmpdir_factory, default_raw_data):
    lt_ctx = lt.Context(executor=InlineJobExecutor())
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-direct'
    default_raw_data.tofile(str(filename))
    del default_raw_data

    ds = lt_ctx.load(
        "raw",
        path=str(filename),
        dtype="float32",
        nav_shape=(16, 16),
        sig_shape=(128, 128),
        io_backend=DirectBackend(),
    )
    yield ds


@pytest.fixture(scope='session')
def big_endian_raw(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-default-big-endian'
    data = utils._mk_random(size=(16, 16, 128, 128), dtype='>u2')
    data.tofile(str(filename))
    del data
    ds = RawFileDataSet(
        path=str(filename),
        nav_shape=(16, 16),
        dtype=">u2",
        sig_shape=(128, 128),
    )
    ds.set_num_cores(2)
    ds = ds.initialize(InlineJobExecutor())
    yield ds


@pytest.fixture(scope='session')
def large_raw_file(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-large-sparse'
    shape = (100, 100, 1216, 1216)
    dtype = np.uint16
    size = np.prod(np.int64(shape)) * np.dtype(dtype).itemsize
    if platform.system() == "Windows":
        os.system(f'FSUtil File CreateNew "{filename}" 0x{size:X}')
        os.system('FSUtil Sparse SetFlag "%s"' % filename)
        os.system(f'FSUtil Sparse SetRange "{filename}" 0 0x{size:X}')
    else:
        with open(filename, 'wb') as f:
            f.truncate(size)
        stat = os.stat(filename)
        if stat.st_blocks != 0:
            warnings.warn(
                f"Created file {filename} is not reported as "
                f"sparse: {stat}, blocks {stat.st_blocks}"
            )
    yield filename, shape, dtype


@pytest.fixture(scope='session')
def large_raw(large_raw_file):
    filename, shape, dtype = large_raw_file
    ds = RawFileDataSet(
        path=str(filename),
        nav_shape=shape[:2],
        dtype=dtype,
        sig_shape=shape[2:],
    )
    ds = ds.initialize(InlineJobExecutor())
    yield ds


@pytest.fixture(scope='session')
def medium_raw_file(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-medium-sparse'
    shape = (128, 128, 256, 256)
    dtype = np.uint16
    size = np.prod(np.int64(shape)) * np.dtype(dtype).itemsize
    if platform.system() == "Windows":
        os.system(f'FSUtil File CreateNew "{filename}" 0x{size:X}')
        os.system('FSUtil Sparse SetFlag "%s"' % filename)
        os.system(f'FSUtil Sparse SetRange "{filename}" 0 0x{size:X}')
    else:
        with open(filename, 'wb') as f:
            f.truncate(size)
        stat = os.stat(filename)
        if stat.st_blocks != 0:
            warnings.warn(
                f"Created file {filename} is not reported as "
                f"sparse: {stat}, blocks {stat.st_blocks}"
            )
    yield filename, shape, dtype


@pytest.fixture(scope='session')
def medium_raw(medium_raw_file):
    filename, shape, dtype = medium_raw_file
    ds = RawFileDataSet(
        path=str(filename),
        nav_shape=shape[:2],
        dtype=dtype,
        sig_shape=shape[2:],
        io_backend=MMapBackend()
    )
    ds = ds.initialize(InlineJobExecutor())
    yield ds


@pytest.fixture(scope='session')
def medium_raw_file_float32(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-medium-sparse'
    shape = (128, 128, 256, 256)
    dtype = np.float32
    size = np.prod(np.int64(shape)) * np.dtype(dtype).itemsize
    if platform.system() == "Windows":
        os.system(f'FSUtil File CreateNew "{filename}" 0x{size:X}')
        os.system('FSUtil Sparse SetFlag "%s"' % filename)
        os.system(f'FSUtil Sparse SetRange "{filename}" 0 0x{size:X}')
    else:
        with open(filename, 'wb') as f:
            f.truncate(size)
        stat = os.stat(filename)
        if stat.st_blocks != 0:
            warnings.warn(
                f"Created file {filename} is not reported as "
                f"sparse: {stat}, blocks {stat.st_blocks}"
            )
    yield filename, shape, dtype


@pytest.fixture(scope='session')
def medium_raw_float32(medium_raw_file_float32):
    filename, shape, dtype = medium_raw_file_float32
    ds = RawFileDataSet(
        path=str(filename),
        nav_shape=shape[:2],
        dtype=dtype,
        sig_shape=shape[2:],
        io_backend=MMapBackend()
    )
    ds = ds.initialize(InlineJobExecutor())
    yield ds


@pytest.fixture(scope='session')
def uint16_raw(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-default-uint16'
    data = utils._mk_random(size=(16, 16, 128, 128), dtype='uint16')
    data.tofile(str(filename))
    del data
    ds = RawFileDataSet(
        path=str(filename),
        nav_shape=(16, 16),
        dtype="uint16",
        sig_shape=(128, 128),
    )
    ds = ds.initialize(InlineJobExecutor())
    yield ds


@pytest.fixture(scope='session')
def raw_with_zeros(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-with-zeros'
    data = np.zeros((16, 16, 128, 128), dtype='float32')
    data.tofile(str(filename))
    del data
    ds = RawFileDataSet(
        path=str(filename),
        nav_shape=(16, 16),
        dtype="float32",
        sig_shape=(128, 128),
    )
    ds.set_num_cores(2)
    ds = ds.initialize(InlineJobExecutor())
    yield ds


@pytest.fixture(scope='session')
def raw_data_8x8x8x8_path(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/8x8x8x8'
    data = utils._mk_random(size=(8, 8, 8, 8), dtype='float32')
    data.tofile(str(filename))
    del data
    yield str(filename)


@pytest.fixture(scope="session")
def mock_sparse_data():
    data = utils._mk_random((13, 17, 24, 19), array_backend=NUMPY)
    data_flat = data.reshape((13*17, 24*19))
    orig = for_backend(data_flat, SCIPY_CSR)
    return orig, data_flat


@pytest.fixture(scope="session")
def raw_csr_generated(mock_sparse_data: tuple[csr_matrix, np.ndarray], tmpdir_factory):
    orig, data_flat = mock_sparse_data
    datadir = tmpdir_factory.mktemp('raw_csr')
    name_indptr = str(datadir / 'indptr.raw')
    name_coords = str(datadir / 'coords.raw')
    name_values = str(datadir / 'values.raw')
    name_sidecar = str(datadir / 'sparse.toml')
    with open(name_sidecar, "w") as f:
        f.write(f"""
[params]
filetype = "raw_csr"
nav_shape = [13, 17]
sig_shape = [24, 19]

[raw_csr]
indptr_file = "indptr.raw"
indptr_dtype = "{str(orig.indptr.dtype)}"

indices_file = "coords.raw"
indices_dtype = "{str(orig.indices.dtype)}"

data_file = "values.raw"
data_dtype = "{str(orig.data.dtype)}"
""")
    orig.indptr.tofile(name_indptr)
    orig.indices.tofile(name_coords)
    orig.data.tofile(name_values)
    lt_ctx = lt.Context.make_with('inline')
    yield lt_ctx.load("raw_csr", path=name_sidecar)


@pytest.fixture(scope="session")
def raw_csr_generated_bigendian(
        mock_sparse_data: tuple[csr_matrix, np.ndarray], tmpdir_factory):
    orig, data_flat = mock_sparse_data
    dtype = np.dtype('>i4')
    datadir = tmpdir_factory.mktemp('raw_csr')
    name_indptr = str(datadir / 'indptr_bigendian.raw')
    name_coords = str(datadir / 'coords_bigendian.raw')
    name_values = str(datadir / 'values_bigendian.raw')
    name_sidecar = str(datadir / 'sparse_bigendian.toml')
    with open(name_sidecar, "w") as f:
        f.write(f"""
[params]
filetype = "raw_csr"
nav_shape = [13, 17]
sig_shape = [24, 19]

[raw_csr]
indptr_file = "indptr_bigendian.raw"
indptr_dtype = "{str(dtype)}"

indices_file = "coords_bigendian.raw"
indices_dtype = "{str(dtype)}"

data_file = "values_bigendian.raw"
data_dtype = "{str(dtype)}"
""")
    orig.indptr.astype(dtype).tofile(name_indptr)
    orig.indices.astype(dtype).tofile(name_coords)
    orig.data.astype(dtype).tofile(name_values)
    lt_ctx = lt.Context.make_with('inline')
    yield lt_ctx.load("raw_csr", path=name_sidecar)


@pytest.fixture(scope="session")
def raw_csr_generated_uint64(mock_sparse_data: tuple[csr_matrix, np.ndarray], tmpdir_factory):
    orig, data_flat = mock_sparse_data
    datadir = tmpdir_factory.mktemp('raw_csr')
    name_indptr = str(datadir / 'indptr_uint64.raw')
    name_coords = str(datadir / 'coords_uint64.raw')
    name_values = str(datadir / 'values_uint64.raw')
    name_sidecar = str(datadir / 'sparse_uint64.toml')
    with open(name_sidecar, "w") as f:
        f.write("""
[params]
filetype = "raw_csr"
nav_shape = [13, 17]
sig_shape = [24, 19]

[raw_csr]
indptr_file = "indptr_uint64.raw"
indptr_dtype = "<u8"

indices_file = "coords_uint64.raw"
indices_dtype = "<u8"

data_file = "values_uint64.raw"
data_dtype = "<u8"
""")
    orig.indptr.astype("<u8").tofile(name_indptr)
    orig.indices.astype("<u8").tofile(name_coords)
    orig.data.astype("<u8").tofile(name_values)
    lt_ctx = lt.Context.make_with('inline')
    yield lt_ctx.load("raw_csr", path=name_sidecar)


@pytest.fixture(scope='session')
def npy_datadir(tmpdir_factory):
    yield tmpdir_factory.mktemp('data_npy')


@pytest.fixture(scope='session')
def default_npy_filepath(npy_datadir):
    yield str(npy_datadir + '/test_default.npy')


@pytest.fixture()
def npy_random_array(npy_datadir):
    random_filename = str(npy_datadir + f'/array{np.random.randint(0, 1000)}.npy')
    ndim = np.random.randint(1, 6)
    shape = tuple(np.random.randint(1, 10) for _ in range(ndim))
    dtype = np.random.choice([np.float32, np.uint8, np.int64, np.complex128])
    array = np.empty(shape, dtype=dtype)
    np.save(random_filename, array)
    return random_filename, array


@pytest.fixture()
def npy_fortran_array(npy_datadir):
    random_filename = str(npy_datadir + f'/array{np.random.randint(0, 1000)}.npy')
    array = np.ones((55, 55), order='F')
    np.save(random_filename, array)
    return random_filename, array


@pytest.fixture(scope='session')
def npy_8x8x8x8_path(npy_datadir):
    filename = npy_datadir + '/8x8x8x8.npy'
    data = utils._mk_random(size=(8, 8, 8, 8), dtype='float32')
    np.save(str(filename), data)
    del data
    yield str(filename)


@pytest.fixture(scope='session')
def npy_8x8x8x8_ds(npy_8x8x8x8_path):
    lt_ctx = lt.Context.make_with('inline')
    ds = lt_ctx.load(
            'npy',
            path=npy_8x8x8x8_path,
        )
    ds.set_num_cores(2)
    yield ds


@pytest.fixture(scope='session')
def default_npy(default_npy_filepath, default_raw_data):
    lt_ctx = lt.Context.make_with('inline')
    filename = default_npy_filepath
    np.save(filename, default_raw_data)
    ds = lt_ctx.load(
        "npy",
        path=filename,
        sig_dims=2,
        io_backend=MMapBackend(),
    )
    ds.set_num_cores(2)
    del default_raw_data
    yield ds


@pytest.fixture
def naughty_filename():
    '''
    Return a string with many special charaters that tests the limits
    of what the file system allows on that platform. This allows to stress-test
    globs or regular expressions applied to paths.
    '''
    system = platform.system()
    # See https://en.wikipedia.org/wiki/Filename#Comparison_of_filename_limitations
    if system == 'Windows':  # NTFS
        return "!§$&[%)(]=`´';,.#~ 🤪"
    elif system == 'Darwin':  # HFS+, APFS?
        return "!\"\\§$&[%)(]=?`´';,.# ~ * | < ** > 🤪"
    else:  # Linux, other Unix
        return "!\"\\§$&[%)(]=?`´':;,.# ~ * | < ** > 🤪"


@pytest.fixture
def scheduler_addr():
    return os.environ['DASK_SCHEDULER_ADDRESS']


@pytest.fixture
def dist_ctx(scheduler_addr):
    """
    This Context needs to have an external dask cluster running, with the following
    assumptions:

     - two workers: hostnames worker-1 and worker-2
     - one scheduler node
     - data availability TBD
    """
    executor = DaskJobExecutor.connect(scheduler_addr)
    with lt.Context(executor=executor) as ctx:
        yield ctx


@pytest.fixture
def ipy_ctx():
    import ipyparallel
    client = ipyparallel.Client()
    # wait for two engines: see also docker-compose.yml where the engines are started
    client.wait_for_engines(2)
    dask_client = client.become_dask()
    executor = DaskJobExecutor(client=dask_client, is_local=False)
    with lt.Context(executor=executor) as ctx:
        yield ctx


# Starting fresh distributed executors takes a lot of time and therefore
# they should be used repeatedly if possible.
# However, some benchmarks require a fresh distributed executor
# and running several Dask executors in parallel has led to lockups when closing
# in some instances.
# That means any shared executor should be shut down before a fresh one is started.
# For that reason we use a fixture with scope "class" and group
# tests in a class that should all use the same executor.
# That way we make sure the shared executor is torn down before any other test
# starts a new one.

# Different from the local_cluster_ctx fixture that only uses two CPUs and at most
# one GPU, this fixture starts a cluster for benchmarking under production condition that
# uses all available CPUs and GPUs. Furthermore, the LiberTEM Context and not only the
# Dask cluster is shared between functions.


@pytest.fixture(scope="class")
def shared_dist_ctx():
    print("start shared Context()")
    ctx = lt.Context()
    yield ctx
    print("stop shared Context()")
    ctx.close()


@pytest.fixture(scope="class")
def shared_dist_ctx_globaldask():
    # Sets default dask.distributed client
    # for integration testing
    print("start shared Context()")
    ctx = lt.Context.make_with('dask-make-default')
    yield ctx
    print("stop shared Context()")
    # Make sure everything is shut down
    ctx.executor.is_local = True
    ctx.close()


@pytest.fixture(autouse=True)
def auto_ctx(doctest_namespace):
    ctx = lt.Context(executor=InlineJobExecutor())
    doctest_namespace["ctx"] = ctx


@pytest.fixture(autouse=True)
def auto_ds(doctest_namespace):
    dataset = MemoryDataSet(datashape=[16, 16, 32, 32])
    doctest_namespace["dataset"] = dataset


@pytest.fixture(autouse=True)
def auto_libs(doctest_namespace):
    doctest_namespace["np"] = np


@pytest.fixture(autouse=True)
def auto_libertem(doctest_namespace):
    import libertem
    import libertem.utils
    import libertem.utils.generate
    import libertem.masks
    import libertem.api
    doctest_namespace["libertem"] = libertem
    doctest_namespace["libertem.utils"] = libertem.utils
    doctest_namespace["libertem.utils.generate"] = libertem.utils.generate
    doctest_namespace["libertem.masks"] = libertem.masks
    doctest_namespace["libertem.api"] = libertem.api


@pytest.fixture(autouse=True)
def auto_files(doctest_namespace, hdf5, default_raw):
    doctest_namespace["path_to_hdf5"] = hdf5.filename
    doctest_namespace["path_to_raw"] = default_raw._path


@pytest.fixture
def inline_executor():
    return InlineJobExecutor(debug=True, inline_threads=2)


@pytest.fixture
def delayed_executor():
    return DelayedJobExecutor()


@pytest.fixture
def lt_ctx(inline_executor):
    return lt.Context(executor=inline_executor, plot_class=Dummy2DPlot)


@pytest.fixture
def inline_executor_fast():
    return InlineJobExecutor(debug=False, inline_threads=2)


@pytest.fixture
def lt_ctx_fast(inline_executor_fast):
    return lt.Context(executor=inline_executor_fast, plot_class=Dummy2DPlot)


@pytest.fixture
async def async_executor(local_cluster_url):
    pool = AsyncAdapter.make_pool()
    sync_executor = await sync_to_async(
        partial(
            DaskJobExecutor.connect,
            scheduler_uri=local_cluster_url
        ),
        pool=pool,
    )
    executor = AsyncAdapter(wrapped=sync_executor, pool=pool)
    yield executor
    await executor.close()


@pytest.fixture
def dask_executor(local_cluster_url):
    executor = DaskJobExecutor.connect(local_cluster_url)
    yield executor
    executor.close()


@pytest.fixture
def concurrent_executor():
    executor = ConcurrentJobExecutor.make_local()
    yield executor
    executor.close()


@pytest.fixture
def delayed_ctx(delayed_executor):
    return lt.Context(executor=delayed_executor, plot_class=Dummy2DPlot)


@pytest.fixture
async def http_client():
    # FIXME: maybe re-scope to module, but would also need
    # adjusted event_loop scope. if we have many API tests
    # maybe reconsider.
    # The timeout needs to be this high to acommodate overloaded
    # CI environments, or otherwise oversubscribed systems
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=120)
    ) as session:
        yield session


@pytest.fixture
async def shared_state():
    event_bus = EventBus()
    executor_state = ExecutorState(event_bus=event_bus)
    yield SharedState(executor_state=executor_state)
    executor_state.shutdown()


class ServerThread(threading.Thread):
    def __init__(self, port, shared_state, token, *args, **kwargs):
        super().__init__(name='LiberTEM-background', *args, **kwargs)
        self.stop_event = threading.Event()
        self.start_event = threading.Event()
        self.port = port
        self.shared_state = shared_state
        self.token = token
        self.loop = None

    async def stop(self):
        self.server.stop()
        await self.server.close_all_connections()
        exstate = self.shared_state.executor_state
        if exstate.have_executor():
            executor = await exstate.get_executor()
            await executor.close()
        self.loop.stop()

    async def wait_for_stop(self):
        """
        background task to periodically check if the main thread wants
        us to stop
        """
        try:
            while True:
                if self.stop_event.is_set():
                    await self.stop()
                    break
                await asyncio.sleep(0.1)
        except Exception:
            await self.stop()
            raise

    def run(self):
        try:
            adjust_event_loop_policy()
            self.loop = loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.set_debug(True)

            event_registry = EventRegistry()
            app = make_app(event_registry, self.shared_state, self.token)
            self.server = tornado.httpserver.HTTPServer(app)
            self.server.listen(address="127.0.0.1", port=self.port)
            # self.shared_state.set_server(self.server)

            asyncio.ensure_future(self.wait_for_stop())
            self.start_event.set()
            loop.run_forever()
        finally:
            self.loop.stop()


@contextlib.contextmanager
def common_server_startup(unused_tcp_port_factory, shared_state, token):
    """
    start a LiberTEM API server on a unused port
    """
    port = unused_tcp_port_factory()

    print(f"starting server at port {port}")
    thread = ServerThread(port, shared_state, token, daemon=True)
    thread.start()
    assert thread.start_event.wait(timeout=5), "server thread failed to start"
    yield port
    print(f"stopping server at port {port}")
    thread.stop_event.set()
    thread.join(timeout=15)
    if thread.is_alive():
        raise RuntimeError("thread did not stop in the given timeout")


@pytest.fixture(scope="function")
def default_token():
    token = "something_random"
    return token


@pytest.fixture(scope="function")
def server_port(unused_tcp_port_factory, shared_state, default_token):
    with common_server_startup(unused_tcp_port_factory, shared_state, token=default_token) as port:
        yield port


@pytest.fixture(scope="function")
async def base_url(server_port):
    return "http://127.0.0.1:%d" % server_port


@pytest.fixture(scope="function")
async def base_url_no_token(unused_tcp_port_factory, shared_state):
    with common_server_startup(
            unused_tcp_port_factory, shared_state, token=None
    ) as server_port:
        yield "http://127.0.0.1:%d" % server_port


def find_unused_port():
    with contextlib.closing(socket.socket()) as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


if sys.platform.startswith("win"):
    # Py3.8+ on Windows need some help with the event loop policy:
    @pytest.fixture(scope="session", autouse=True)
    def event_loop_policy():
        # see also: libertem.common.async_utils.adjust_event_loop_policy
        try:
            from asyncio import (
                WindowsProactorEventLoopPolicy,
                WindowsSelectorEventLoopPolicy,
            )
        except ImportError:
            return asyncio.get_event_loop_policy()
        else:
            # FIXME this might fail if the event loop policy has been overridden by something
            # incompatible that is not a WindowsProactorEventLoopPolicy. Dask.distributed creates
            # a custom event loop policy, for example. Fortunately that one should be compatible.
            # Better would be "duck typing", i.e. to check if the required add_reader() method
            # of the loop throws a NotImplementedError.
            if type(asyncio.get_event_loop_policy()) is WindowsProactorEventLoopPolicy:
                # WindowsProactorEventLoopPolicy is not compatible with tornado 6
                # fallback to the pre-3.8 default of Selector
                return WindowsSelectorEventLoopPolicy()


def _local_cluster_url():
    """
    Shared dask cluster, can be used repeatedly by different executors.

    This allows numba caching across tests, without sharing the executor,
    for example
    """
    cluster_port = find_unused_port()
    devices = detect()
    spec = cluster_spec(
        # Only use at most 2 CPUs and 1 GPU
        cpus=devices['cpus'][:2],
        cudas=devices['cudas'][:1],
        has_cupy=devices['has_cupy']
    )

    cluster_kwargs = {
        'silence_logs': logging.WARN,
        'scheduler': {
            'cls': Scheduler,
            'options': {'port': cluster_port},
        },
    }

    # use the context manager of SpecCluster here to properly
    # shut down the LoopRunner in case of older distributed versions:
    with dd.SpecCluster(
        workers=spec,
        **(cluster_kwargs or {})
    ) as cluster:
        with set_num_threads_env(1, set_numba=False):
            client = dd.Client(cluster, set_as_default=False)
            client.wait_for_workers(len(spec))
            client.close()

        yield 'tcp://localhost:%d' % cluster_port


local_cluster_url = pytest.fixture(scope='session')(_local_cluster_url)
local_cluster_url_per_module = pytest.fixture(scope='module')(_local_cluster_url)


@pytest.fixture(scope='function')
def local_cluster_ctx(dask_executor):
    return lt.Context(executor=dask_executor)


@pytest.fixture
def concurrent_ctx(concurrent_executor):
    return lt.Context(executor=concurrent_executor)


@pytest.fixture(scope='session')
def pipelined_ctx():
    # limit the number of CPU workers started by default
    with lt.Context.make_with('pipelined', cpus=2) as ctx:
        yield ctx


@pytest.fixture
def points():
    return np.array([
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (0, -1),
        (-1, 0),
        (-1, -1)
    ])


@pytest.fixture(scope="function")
def standard_bad_ds_params():
    return [
        dict(nav_shape=(-1, 2)),
        dict(nav_shape=4),
        dict(sig_shape=(3, 17)),
        dict(sig_shape=(103, -83)),
        dict(sync_offset=180000000),
        dict(io_backend="doesn't exist"),
    ]


@pytest.fixture()
def ds_params_tester(lt_ctx, bad_params_logger):
    def _test_bad_params(ds_key, *args, **params):
        ds_class = get_dataset_cls(ds_key)
        kwarg_repr = (
            textwrap.shorten(f'{k}={repr(v)}', 30, placeholder="...")
            for k, v in params.items()
        )
        stub = (
            f"{ds_class.__name__}(..., {', '.join(kwarg_repr)})"
        )
        try:
            lt_ctx.load(ds_key, *args, **params)
        except Exception as e:
            bad_params_logger.append(f'{stub}\n  RAISED {type(e).__name__}("{e}")')
            return False
        bad_params_logger.append(f'{stub}\n  DID NOT raise')
        return True
    return _test_bad_params


@pytest.fixture(scope="session")
def bad_params_logger():
    lines = []
    yield lines
    print("\n".join(lines))


@pytest.hookimpl(hookwrapper=True)
def pytest_benchmark_generate_json(config, benchmarks, include_data, machine_info, commit_info):
    machine_info["freeze"] = [
        (d.metadata['Name'], d.version)
        for d in distributions()
    ]
    yield


def pytest_collectstart(collector):
    # nbval: ignore some output types
    if collector.fspath and collector.fspath.ext == '.ipynb':
        collector.skip_compare += 'text/html', 'application/javascript', 'stderr',
