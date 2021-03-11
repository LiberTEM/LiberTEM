import time
import asyncio

import pytest

from libertem.executor.base import async_generator_eager, async_generator


@pytest.mark.asyncio
async def test_async_generator_eager():
    def sync_generator():
        t0 = time.time()
        for i in range(2):
            time.sleep(0.1)
            yield 'result'
        print(time.time() - t0)

    t0 = time.time()

    gen = sync_generator()
    async_gen = async_generator_eager(gen)

    async for value in async_gen:
        assert value == 'result'
        await asyncio.sleep(0.1)  # simulate something else happening in the main thread

    t1 = time.time()
    delta = t1 - t0

    assert 0.3 <= delta < 0.4

    t0 = time.time()

    gen = sync_generator()
    async_gen = async_generator(gen)

    async for value in async_gen:
        assert value == 'result'
        await asyncio.sleep(0.1)  # simulate something else happening in the main thread

    t1 = time.time()
    delta_lazy = t1 - t0

    assert 0.4 <= delta_lazy < 0.5

    assert delta < delta_lazy


@pytest.mark.asyncio
async def test_async_generator_err_handling():
    def sync_generator():
        t0 = time.time()
        for i in range(2):
            time.sleep(0.01)
            yield 'result'
        raise RuntimeError("something unexpected happened")
        print(time.time() - t0)

    gen = sync_generator()
    async_gen = async_generator_eager(gen)

    with pytest.raises(RuntimeError) as e:
        async for value in async_gen:
            assert value == 'result'

    assert e.match("something unexpected happened")
