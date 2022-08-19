import time
import concurrent

import pytest
import numpy as np

from libertem.common.async_utils import async_generator_eager, async_generator
from libertem.common.executor import SimpleMPWorkerQueue, SimpleWorkerQueue
from libertem.common.buffers import reshaped_view


@pytest.fixture(scope='module')
def thread_pool():
    pool = concurrent.futures.ThreadPoolExecutor(1)
    # warm up:
    pool.submit(lambda: 1).result()
    return pool


@pytest.mark.asyncio
async def test_async_generator_eager(thread_pool):
    results = []

    def sync_generator():
        for i in range(20):
            results.append('result')
            yield 'result'

    gen = sync_generator()
    async_gen = async_generator_eager(gen, pool=thread_pool)

    await async_gen.__anext__()
    time.sleep(0.1)

    # the generator has run to completion already:
    assert len(results) == 20

    async for value in async_gen:
        assert value == 'result'

    results = []

    gen = sync_generator()
    async_gen = async_generator(gen, pool=thread_pool)

    await async_gen.__anext__()
    time.sleep(0.1)

    # the generator has only run a single iteration:
    assert len(results) == 1

    async for value in async_gen:
        assert value == 'result'


@pytest.mark.asyncio
async def test_async_generator_err_handling(thread_pool):
    def sync_generator():
        for i in range(2):
            yield 'result'
        raise RuntimeError("something unexpected happened")

    gen = sync_generator()
    async_gen = async_generator_eager(gen, pool=thread_pool)

    with pytest.raises(RuntimeError) as e:
        async for value in async_gen:
            assert value == 'result'

    assert e.match("something unexpected happened")


@pytest.mark.parametrize(
    'queue', [SimpleMPWorkerQueue(), SimpleWorkerQueue()]
)
def test_worker_queues(queue):
    payload_data = np.random.random((23, 42))
    header = {
        'asdf': lambda x: x,
        'abc': 23,
        'shape': payload_data.shape,
        'dtype': payload_data.dtype,
    }

    def check_received(received_header, decoded_payload):
        assert received_header['asdf']('abc') == 'abc'
        for key in ('abc', 'shape', 'dtype'):
            assert header[key] == received_header[key]
        assert np.all(payload_data == decoded_payload)

    queue.put(header, payload_data)
    with queue.get() as res:
        check_received(*res)

    with queue.put_nocopy(header, size=payload_data.nbytes) as payload:
        payload[:] = reshaped_view(payload_data, (23*42, )).view(np.uint8)

    with queue.get() as res:
        received_header, received_payload = res
        decoded_payload = reshaped_view(
            np.asarray(received_payload).view(received_header['dtype']),
            received_header['shape']
        )
        check_received(received_header, decoded_payload)
