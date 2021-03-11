import time
import asyncio
import concurrent

import pytest

from libertem.executor.base import async_generator_eager, async_generator


@pytest.fixture(scope='module')
def thread_pool():
    pool = concurrent.futures.ThreadPoolExecutor(1)
    # warm up:
    pool.submit(lambda: 1).result()
    return pool


@pytest.mark.asyncio
async def test_async_generator_eager(thread_pool):
    def sync_generator():
        t0 = time.time()
        for i in range(2):
            time.sleep(0.1)
            yield 'result'
        print(time.time() - t0)

    t0 = time.time()

    gen = sync_generator()
    async_gen = async_generator_eager(gen, pool=thread_pool)

    async for value in async_gen:
        assert value == 'result'
        await asyncio.sleep(0.1)  # simulate something else happening in the main thread

    t1 = time.time()
    delta = t1 - t0

    assert 0.3 <= delta < 0.4

    t0 = time.time()

    gen = sync_generator()
    async_gen = async_generator(gen, pool=thread_pool)

    async for value in async_gen:
        assert value == 'result'
        await asyncio.sleep(0.1)  # simulate something else happening in the main thread

    t1 = time.time()
    delta_lazy = t1 - t0

    assert 0.4 <= delta_lazy < 0.5

    assert delta < delta_lazy


@pytest.mark.asyncio
async def test_async_generator_err_handling(thread_pool):
    def sync_generator():
        t0 = time.time()
        for i in range(2):
            time.sleep(0.01)
            yield 'result'
        raise RuntimeError("something unexpected happened")
        print(time.time() - t0)

    gen = sync_generator()
    async_gen = async_generator_eager(gen, pool=thread_pool)

    with pytest.raises(RuntimeError) as e:
        async for value in async_gen:
            assert value == 'result'

    assert e.match("something unexpected happened")
