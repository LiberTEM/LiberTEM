import os
import pytest


@pytest.mark.asyncio
async def test_browse_localfs(default_raw, base_url, http_client):
    conn_url = "{}/api/config/connection/".format(base_url)
    conn_details = {
        'connection': {
            'type': 'local',
            'numWorkers': 2,
        }
    }
    async with http_client.put(conn_url, json=conn_details) as response:
        assert response.status == 200

    browse_path = os.path.dirname(default_raw._path)
    raw_ds_filename = os.path.basename(default_raw._path)
    url = "{}/api/browse/localfs/".format(base_url)
    async with http_client.get(url, params={"path": browse_path}) as resp:
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
            assert set(entry.keys()) == set(["name", "size", "ctime", "mtime", "owner"])
            if entry["name"] == raw_ds_filename:
                defraw_found = True
            assert defraw_found


@pytest.mark.asyncio
async def test_load_1(default_raw, base_url, http_client):
    conn_url = "{}/api/config/connection/".format(base_url)
    conn_details = {
        'connection': {
            'type': 'local',
            'numWorkers': 2,
        }
    }
    async with http_client.put(conn_url, json=conn_details) as response:
        assert response.status == 200

    raw_path = default_raw._path

    uuid = "ae5d23bd-1f2a-4c57-bab2-dfc59a1219f3"
    ds_url = "{}/api/datasets/{}/".format(
        base_url, uuid
    )
    ds_data = {
        "dataset": {
            "params": {
                "type": "raw",
                "path": raw_path,
                "dtype": "float32",
                "detector_size_raw": [128, 128],
                "crop_detector_to": [128, 128],
                "tileshape": [1, 1, 128, 128],
                "scan_size": [16, 16]
            }
        }
    }
    async with http_client.put(ds_url, json=ds_data) as resp:
        assert resp.status == 200
        resp_json = await resp.json()
        assert resp_json['status'] == 'ok'
        for k in ds_data['dataset']['params']:
            assert ds_data['dataset']['params'][k] == resp_json['details']['params'][k]
