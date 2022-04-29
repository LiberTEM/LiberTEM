import time
import concurrent

import pytest

from libertem.common.async_utils import async_generator_eager, async_generator


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
