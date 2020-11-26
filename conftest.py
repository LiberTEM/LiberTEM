import asyncio
import os
import time
import importlib.util
import platform
import threading
import pkg_resources
from functools import partial
import warnings

import numpy as np
import pytest
import h5py
import aiohttp

import libertem.api as lt
from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.hdf5 import H5DataSet
from libertem.io.dataset.raw import RawFileDataSet
from libertem.io.dataset.memory import MemoryDataSet
from libertem.executor.dask import DaskJobExecutor, cluster_spec

from libertem.web.server import make_app, EventRegistry
from libertem.web.state import SharedState
from libertem.executor.base import AsyncAdapter, sync_to_async
from libertem.utils.async_utils import adjust_event_loop_policy

# A bit of gymnastics to import the test utilities since this
# conftest.py file is shared between the doctests and unit tests
# and this file is outside the package
basedir = os.path.dirname(__file__)
location = os.path.join(basedir, "tests/utils.py")
spec = importlib.util.spec_from_file_location("utils", location)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


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
def hdf5_3d(tmpdir_factory):
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-3d.h5", data=np.ones((17, 16, 16)))


@pytest.fixture(scope='session')
def hdf5_5d(tmpdir_factory):
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-5d.h5",
                                  data=np.ones((3, 5, 9, 16, 16)))


@pytest.fixture(scope='session')
def random_hdf5(tmpdir_factory):
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-random.h5",
                                  data=np.random.randn(5, 5, 16, 16))


@pytest.fixture(scope='session')
def chunked_hdf5(tmpdir_factory):
    yield from get_or_create_hdf5(tmpdir_factory, "hdf5-test-chunked.h5",
                                  data=np.ones((5, 5, 16, 16)),
                                  chunks=(1, 2, 16, 16))


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
def hdf5_ds_3d(hdf5_3d):
    ds = H5DataSet(
        path=hdf5_3d.filename, ds_path="data",
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
def default_raw(tmpdir_factory, default_raw_data):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-default'
    default_raw_data.tofile(str(filename))
    del default_raw_data
    ds = RawFileDataSet(
        path=str(filename),
        scan_size=(16, 16),
        dtype="float32",
        detector_size=(128, 128),
    )
    ds.set_num_cores(2)
    ds = ds.initialize(InlineJobExecutor())
    yield ds


@pytest.fixture(scope='session')
def big_endian_raw(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-test-default'
    data = utils._mk_random(size=(16, 16, 128, 128), dtype='>u2')
    data.tofile(str(filename))
    del data
    ds = RawFileDataSet(
        path=str(filename),
        scan_size=(16, 16),
        dtype=">u2",
        detector_size=(128, 128),
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
        os.system('FSUtil File CreateNew "%s" 0x%X' % (filename, size))
        os.system('FSUtil Sparse SetFlag "%s"' % filename)
        os.system('FSUtil Sparse SetRange "%s" 0 0x%X' % (filename, size))
    else:
        with open(filename, 'wb') as f:
            f.truncate(size)
        stat = os.stat(filename)
        if stat.st_blocks != 0:
            warnings.warn(f"Created file {filename} is not reported as sparse: {stat}, blocks {stat.st_blocks}")
    yield filename, shape, dtype


@pytest.fixture(scope='session')
def large_raw(large_raw_file):
    filename, shape, dtype = large_raw_file
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


@pytest.fixture(scope='session')
def raw_with_zeros(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/raw-with-zeros'
    data = np.zeros((16, 16, 128, 128), dtype='float32')
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


@pytest.fixture
def ipy_ctx():
    import ipyparallel
    client = ipyparallel.Client()
    retries = 10
    while retries > 0:
        retries -= 1
        if len(client.ids) > 0:
            break
        time.sleep(1)
    dask_client = client.become_dask()
    executor = DaskJobExecutor(client=dask_client, is_local=False)
    with lt.Context(executor=executor) as ctx:
        yield ctx


# Starting fresh distributed executors takes a lot of time and therefore
# they should be used repeatedly if possible.
# However, some benchmarks require a fresh distributed executor
# and running several Dask executors in parallel leads to lockups when closing.
# That means any shared executor has to be shut down before a fresh one is started.
# For that reason we use a fixture with scope "class" and group
# tests in a class that should all use the same executor.
# That way we make sure the shared executor is torn down before any other test
# starts a new one.


@pytest.fixture(scope="class")
def shared_dist_ctx():
    print("start shared Context()")
    ctx = lt.Context()
    yield ctx
    print("stop shared Context()")
    ctx.close()


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


@pytest.fixture
def inline_executor():
    return InlineJobExecutor(debug=True)


@pytest.fixture
def lt_ctx(inline_executor):
    return lt.Context(executor=inline_executor)


@pytest.fixture
async def async_executor():
    spec = cluster_spec(cpus=[0, 1], cudas=[], has_cupy=False)

    pool = AsyncAdapter.make_pool()
    sync_executor = await sync_to_async(partial(DaskJobExecutor.make_local, spec=spec), pool=pool)
    executor = AsyncAdapter(wrapped=sync_executor, pool=pool)
    yield executor
    await executor.close()


@pytest.fixture
def dask_executor():
    sync_executor = DaskJobExecutor.make_local()
    yield sync_executor
    sync_executor.close()


@pytest.fixture
async def http_client():
    # FIXME: maybe re-scope to module, but would also need
    # adjusted event_loop scope. if we have many API tests
    # maybe reconsider.
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=30)
    ) as session:
        yield session


@pytest.fixture
def shared_state():
    return SharedState()


class ServerThread(threading.Thread):
    def __init__(self, port, shared_state, *args, **kwargs):
        super().__init__(name='LiberTEM-background', *args, **kwargs)
        self.stop_event = threading.Event()
        self.start_event = threading.Event()
        self.port = port
        self.shared_state = shared_state
        self.loop = None

    async def stop(self):
        self.server.stop()
        await self.server.close_all_connections()
        if self.shared_state.executor_state.have_executor():
            await self.shared_state.executor_state.get_executor().close()
        self.loop.stop()

    async def wait_for_stop(self):
        """
        background task to periodically check if the main thread wants
        us to stop
        """
        while True:
            if self.stop_event.is_set():
                await self.stop()
                break
            await asyncio.sleep(0.1)

    def run(self):
        try:
            adjust_event_loop_policy()
            self.loop = loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.set_debug(True)

            event_registry = EventRegistry()
            app = make_app(event_registry, self.shared_state)
            self.server = app.listen(address="127.0.0.1", port=self.port)
            # self.shared_state.set_server(self.server)

            asyncio.ensure_future(self.wait_for_stop())
            self.start_event.set()
            loop.run_forever()
        finally:
            self.loop.stop()


@pytest.fixture(scope="function")
def server_port(unused_tcp_port_factory, shared_state):
    """
    start a LiberTEM API server on a unused port
    """
    port = unused_tcp_port_factory()

    print("starting server at port {}".format(port))
    thread = ServerThread(port, shared_state)
    thread.start()
    assert thread.start_event.wait(timeout=1), "server thread failed to start"
    yield port
    print("stopping server at port {}".format(port))
    thread.stop_event.set()
    thread.join(timeout=15)
    if thread.is_alive():
        raise RuntimeError("thread did not stop in the given timeout")


@pytest.fixture(scope="function")
async def base_url(server_port):
    return "http://127.0.0.1:%d" % server_port


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


@pytest.fixture
def indices():
    return np.array([
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
        (-1, 0),
        (0, -1),
        (-1, -1)
    ])


@pytest.fixture
def zero():
    return np.array([0, 0])


@pytest.fixture
def a():
    return np.array([0, 1])


@pytest.fixture
def b():
    return np.array([1, 0])


@pytest.mark.hookwrapper
def pytest_benchmark_generate_json(config, benchmarks, include_data, machine_info, commit_info):
    machine_info["freeze"] = [(d.key, d.version) for d in pkg_resources.working_set]
    yield
