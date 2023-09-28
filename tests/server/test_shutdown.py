import pytest

from utils import assert_msg
from aio_utils import create_connection

pytestmark = [pytest.mark.web_api]


@pytest.mark.asyncio
async def test_shutdown(base_url, http_client, server_port, default_token):
    await create_connection(base_url, http_client, token=default_token)

    print("checkpoint 1")

    url = f"ws://127.0.0.1:{server_port}/api/shutdown/?token={default_token}"
    async with http_client.delete(url) as resp:
        print("checkpoint 2")
        assert resp.status == 200
        assert_msg(await resp.json(), "SERVER_SHUTDOWN")
