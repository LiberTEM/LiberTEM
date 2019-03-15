import json

import pytest
import websockets

from utils import assert_msg


@pytest.mark.asyncio
async def test_detect_failed(default_raw, base_url, http_client, server_port):
    conn_url = "{}/api/config/connection/".format(base_url)
    conn_details = {
        'connection': {
            'type': 'local',
            'numWorkers': 2,
        }
    }
    async with http_client.put(conn_url, json=conn_details) as response:
        assert response.status == 200

    # connect to ws endpoint:
    ws_url = "ws://127.0.0.1:{}/api/events/".format(server_port)
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')
        assert initial_msg['datasets'] == []
        assert initial_msg['jobs'] == []

        path = default_raw._path
        detect_url = "{}/api/datasets/detect/".format(base_url)

        async with http_client.get(detect_url, params={"path": path}) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DATASET_DETECTION_FAILED', status='error')


@pytest.mark.asyncio
async def test_detect_hdf5(hdf5, base_url, http_client, server_port):
    conn_url = "{}/api/config/connection/".format(base_url)
    conn_details = {
        'connection': {
            'type': 'local',
            'numWorkers': 2,
        }
    }
    async with http_client.put(conn_url, json=conn_details) as response:
        assert response.status == 200

    # connect to ws endpoint:
    ws_url = "ws://127.0.0.1:{}/api/events/".format(server_port)
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')
        assert initial_msg['datasets'] == []
        assert initial_msg['jobs'] == []

        path = hdf5.filename
        detect_url = "{}/api/datasets/detect/".format(base_url)

        async with http_client.get(detect_url, params={"path": path}) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DATASET_DETECTED')
            assert "datasetParams" in resp_json
            params = resp_json["datasetParams"]
            assert params['type'] == 'HDF5'
            assert params['path'] == path
            assert params['ds_path'] == "data"
            assert "tileshape" in params
