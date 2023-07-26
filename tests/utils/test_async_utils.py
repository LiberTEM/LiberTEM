import pytest

from libertem.common.async_utils import (
    sync_to_async, MyStopIteration, run_agen_get_last, run_gen_get_last,
)


def _f():
    print("stuff while")
    next(iter([]))


@pytest.mark.asyncio
async def test_sync_to_async_stopiteration():
    print("stuff before")
    with pytest.raises(MyStopIteration):
        await sync_to_async(_f)


@pytest.mark.asyncio
async def test_run_agen_get_last_empty():
    async def _empty_agen():
        for i in []:
            yield i

    with pytest.raises(RuntimeError):
        await run_agen_get_last(_empty_agen())


def test_run_gen_get_last_empty():
    def _empty_gen():
        yield from []

    with pytest.raises(RuntimeError):
        run_gen_get_last(_empty_gen())
