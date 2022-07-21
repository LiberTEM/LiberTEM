import os
import pytest

from aio_utils import create_connection

pytestmark = [pytest.mark.slow]


@pytest.mark.asyncio
async def test_browse_localfs(default_raw, base_url, http_client, local_cluster_url, default_token):
    await create_connection(base_url, http_client, local_cluster_url, default_token)
    browse_path = os.path.dirname(default_raw._path)
    raw_ds_filename = os.path.basename(default_raw._path)
    url = f"{base_url}/api/browse/localfs/"
    async with http_client.get(url, params={"path": browse_path, "token": default_token}) as resp:
        assert resp.status == 200
        listing = await resp.json()
        assert listing['status'] == 'ok'
        assert listing['messageType'] == 'DIRECTORY_LISTING'
        assert "drives" in listing
        assert "places" in listing
        assert "path" in listing
        assert "files" in listing
        assert "dirs" in listing
        assert listing["path"] == browse_path
        assert len(listing["files"]) >= 1
        defraw_found = False
        for entry in listing["files"]:
            assert set(entry.keys()) == {"name", "size", "ctime", "mtime", "owner"}
            if entry["name"] == raw_ds_filename:
                defraw_found = True
            assert defraw_found


@pytest.mark.asyncio
async def test_browse_localfs_fail(
    default_raw, base_url, http_client, local_cluster_url, default_token
):
    await create_connection(base_url, http_client, local_cluster_url, default_token)
    browse_path = os.path.join(
        os.path.dirname(default_raw._path),
        "does", "not", "exist"
    )
    url = f"{base_url}/api/browse/localfs/"
    async with http_client.get(url, params={"path": browse_path, "token": default_token}) as resp:
        assert resp.status == 200
        listing = await resp.json()
        assert listing['status'] == 'error'
        assert listing['messageType'] == 'DIRECTORY_LISTING_FAILED'
