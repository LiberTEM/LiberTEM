import asyncio

import pytest
import h5py
import numpy as np
import aiohttp

from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.hdf5 import H5DataSet
from libertem.io.dataset.raw import RawFileDataSet
from libertem.web.server import make_app, EventRegistry, SharedData
from libertem import api as lt

from utils import MemoryDataSet, _mk_random


@pytest.fixture
def inline_executor():
    return InlineJobExecutor(debug=True)


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
def random_hdf5(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('data')
    filename = datadir + '/hdf5-test-random.h5'
    try:
        with h5py.File(filename, 'r') as f:
            yield f
    except OSError:
        with h5py.File(filename, "w") as f:
            f.create_dataset("data", data=_mk_random(size=(5, 5, 16, 16), dtype="float32"))
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
    data = _mk_random(size=(16, 16, 128, 128), dtype='float32')
    data.tofile(str(filename))
    del data
    ds = RawFileDataSet(
        path=str(filename),
        scan_size=(16, 16),
        dtype="float32",
        detector_size=(128, 128),
    )
    ds = ds.initialize()
    yield ds


@pytest.fixture
async def http_client():
    # FIXME: maybe re-scope to module, but would also need
    # adjusted event_loop scope. if we have many API tests
    # maybe reconsider.
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=10)
    ) as session:
        yield session


@pytest.fixture(scope="function")
async def server_port(unused_tcp_port_factory):
    """
    start a LiberTEM API server on a unused port
    """
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    port = unused_tcp_port_factory()
    event_registry = EventRegistry()
    shared_data = SharedData()
    app = make_app(event_registry, shared_data)
    print("starting server at port {}".format(port))
    server = app.listen(address="127.0.0.1", port=port)
    yield port
    print("stopping server at port {}".format(port))
    server.stop()
    await server.close_all_connections()
    if shared_data.have_executor():
        await shared_data.get_executor().close()


@pytest.fixture(scope="function")
async def base_url(server_port):
    return "http://127.0.0.1:%d" % server_port
