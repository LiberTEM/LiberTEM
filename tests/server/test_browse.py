import os
import pytest

from aio_utils import create_connection

pytestmark = [pytest.mark.web_api]


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
async def test_browse_localfs_stat(
    default_raw, base_url, http_client, local_cluster_url, default_token,
):
    await create_connection(base_url, http_client, local_cluster_url, default_token)
    browse_path = os.path.dirname(default_raw._path)
    raw_ds_filename = os.path.basename(default_raw._path)
    url = f"{base_url}/api/browse/localfs/stat/"

    # stat the directory:
    async with http_client.get(url, params={"path": browse_path, "token": default_token}) as resp:
        assert resp.status == 200
        stat_result = await resp.json()
        assert stat_result['status'] == 'ok'
        assert stat_result['messageType'] == 'STAT_RESULT'
        assert "basename" in stat_result
        assert stat_result["dirname"] == browse_path
        assert stat_result["path"] == browse_path
        for attr in ["size", "ctime", "mtime"]:
            assert attr in stat_result["stat"]
        assert stat_result["stat"]["isdir"]

    # stat the file:
    async with http_client.get(
        url, params={"path": default_raw._path, "token": default_token},
    ) as resp:
        assert resp.status == 200
        stat_result = await resp.json()
        assert stat_result['status'] == 'ok'
        assert stat_result['messageType'] == 'STAT_RESULT'
        assert stat_result["basename"] == raw_ds_filename
        assert stat_result["dirname"] == browse_path
        assert stat_result["path"] == default_raw._path
        for attr in ["size", "ctime", "mtime"]:
            assert attr in stat_result["stat"]
        assert not stat_result["stat"]["isdir"]
        assert stat_result["stat"]["isreg"]


@pytest.mark.asyncio
async def test_browse_localfs_stat_fail(
    default_raw, base_url, http_client, local_cluster_url, default_token,
):
    await create_connection(base_url, http_client, local_cluster_url, default_token)
    url = f"{base_url}/api/browse/localfs/stat/"
    path = '/does/not/really/exist/'

    # stat the directory:
    async with http_client.get(
        url, params={"path": path, "token": default_token}
    ) as resp:
        assert resp.status == 200
        stat_result = await resp.json()
        assert stat_result['status'] == 'error'
        assert stat_result['messageType'] == 'STAT_FAILED'
        assert 'alternative' in stat_result
        assert 'msg' in stat_result
        assert stat_result['path'] == path


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
    async with http_client.get(
        url, params={"path": browse_path, "token": default_token}
    ) as resp:
        assert resp.status == 200
        listing = await resp.json()
        assert listing['status'] == 'error'
        assert listing['messageType'] == 'DIRECTORY_LISTING_FAILED'
