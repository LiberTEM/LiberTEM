import pytest

from utils import assert_msg
from aio_utils import create_connection

pytestmark = [pytest.mark.functional]


@pytest.mark.asyncio
async def test_shutdown(base_url, http_client, server_port):
    await create_connection(base_url, http_client)

    print("checkpoint 1")

    url = f"ws://127.0.0.1:{server_port}/api/shutdown/"
    async with http_client.delete(url) as resp:
        print("checkpoint 2")
        assert resp.status == 200
        assert_msg(await resp.json(), "SERVER_SHUTDOWN")
