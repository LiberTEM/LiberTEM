import asyncio
import functools

import pytest

import numpy as np
import aiohttp

from libertem.executor.inline import InlineJobExecutor

from libertem.web.server import make_app, EventRegistry
from libertem.web.state import SharedState
from libertem.executor.base import AsyncAdapter, sync_to_async
from libertem.executor.dask import DaskJobExecutor, cluster_spec
from libertem import api as lt


@pytest.fixture
def inline_executor():
    return InlineJobExecutor(debug=True)


@pytest.fixture
def lt_ctx(inline_executor):
    return lt.Context(executor=inline_executor)


@pytest.fixture
async def async_executor():
    spec = cluster_spec(cpus=[0, 1], cudas=[])
    sync_executor = await sync_to_async(
        functools.partial(DaskJobExecutor.make_local, spec=spec)
    )
    executor = AsyncAdapter(wrapped=sync_executor)
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
        timeout=aiohttp.ClientTimeout(total=20)
    ) as session:
        yield session


@pytest.fixture
def shared_state():
    return SharedState()


@pytest.fixture(scope="function")
async def server_port(unused_tcp_port_factory, shared_state):
    """
    start a LiberTEM API server on a unused port
    """
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    port = unused_tcp_port_factory()
    event_registry = EventRegistry()
    app = make_app(event_registry, shared_state)
    print("starting server at port {}".format(port))
    server = app.listen(address="127.0.0.1", port=port)
    yield port
    print("stopping server at port {}".format(port))
    server.stop()
    await server.close_all_connections()
    if shared_state.executor_state.have_executor():
        await shared_state.executor_state.get_executor().close()


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
