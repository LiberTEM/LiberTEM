import json

import pytest
import websockets

from utils import assert_msg

pytestmark = [pytest.mark.functional]


def _get_raw_params(path):
    return {
        "dataset": {
            "params": {
                "type": "RAW",
                "path": path,
                "dtype": "float32",
                "detector_size": [128, 128],
                "enable_direct": False,
                "tileshape": [1, 1, 128, 128],
                "scan_size": [16, 16]
            }
        }
    }


@pytest.mark.asyncio
async def test_load_raw_success(default_raw, base_url, http_client):
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
    ds_data = _get_raw_params(raw_path)
    async with http_client.put(ds_url, json=ds_data) as resp:
        assert resp.status == 200
        resp_json = await resp.json()
        assert_msg(resp_json, 'CREATE_DATASET')
        for k in ds_data['dataset']['params']:
            assert ds_data['dataset']['params'][k] == resp_json['details']['params'][k]


@pytest.mark.asyncio
async def test_load_raw_fail(default_raw, base_url, http_client):
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
    ds_data = _get_raw_params(raw_path)
    ds_data["dataset"]["params"]["scan_size"] = [32, 32]  # too large, should cause error
    async with http_client.put(ds_url, json=ds_data) as resp:
        assert resp.status == 200
        resp_json = await resp.json()
        assert_msg(resp_json, 'CREATE_DATASET_ERROR', status='error')
        assert resp_json['dataset'] == uuid
        assert resp_json['msg'].startswith('invalid dataset: ')


@pytest.mark.asyncio
async def test_dataset_delete(default_raw, base_url, http_client, server_port):
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
    ds_data = _get_raw_params(raw_path)

    # connect to ws endpoint:
    ws_url = "ws://127.0.0.1:{}/api/events/".format(server_port)
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        async with http_client.put(ds_url, json=ds_data) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'CREATE_DATASET')

        async with http_client.delete(ds_url) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DELETE_DATASET')


@pytest.mark.asyncio
async def test_initial_state_after_reconnect(default_raw, base_url, http_client, server_port):
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
    ds_data = _get_raw_params(raw_path)

    # connect to ws endpoint:
    ws_url = "ws://127.0.0.1:{}/api/events/".format(server_port)
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        async with http_client.put(ds_url, json=ds_data) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'CREATE_DATASET')

    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')
        assert initial_msg["jobs"] == []
        assert len(initial_msg["datasets"]) == 1
        assert initial_msg["datasets"][0]["id"] == uuid
        assert initial_msg["datasets"][0]["params"] == {
            'detector_size': [128, 128],
            "enable_direct": False,
            'dtype': 'float32',
            'path': raw_path,
            'scan_size': [16, 16],
            'shape': [16, 16, 128, 128],
            'tileshape': [1, 1, 128, 128],
            'type': 'RAW'
        }
        assert len(initial_msg["datasets"][0]["diagnostics"]) == 2
