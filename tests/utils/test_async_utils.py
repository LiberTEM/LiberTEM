import pytest

from libertem.utils.async_utils import sync_to_async, MyStopIteration


def _f():
    print("stuff while")
    next(iter([]))


@pytest.mark.asyncio
async def test_sync_to_async_stopiteration():
    print("stuff before")
    with pytest.raises(MyStopIteration):
        await sync_to_async(_f)
