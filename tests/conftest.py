import asyncio
import functools
import threading

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
