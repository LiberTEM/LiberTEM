import json

import pytest
import websockets

from utils import assert_msg
from aio_utils import create_connection

pytestmark = [pytest.mark.web_api]


@pytest.mark.asyncio
async def test_detect_failed(
    default_raw, base_url, http_client, server_port, local_cluster_url, default_token,
):
    await create_connection(base_url, http_client, local_cluster_url, default_token)
    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/?token={default_token}"
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')
        assert initial_msg['datasets'] == []
        assert initial_msg['jobs'] == []

        path = default_raw._path
        detect_url = f"{base_url}/api/datasets/detect/?token={default_token}"

        async with http_client.get(detect_url, params={"path": path}) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DATASET_DETECTION_FAILED', status='error')


@pytest.mark.asyncio
async def test_detect_hdf5(
    hdf5, base_url, http_client, server_port, local_cluster_url, default_token,
):
    await create_connection(base_url, http_client, local_cluster_url, default_token)
    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/?token={default_token}"
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')
        assert initial_msg['datasets'] == []
        assert initial_msg['jobs'] == []

        path = hdf5.filename
        detect_url = f"{base_url}/api/datasets/detect/?token={default_token}"

        async with http_client.get(detect_url, params={"path": path}) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DATASET_DETECTED')
            assert "datasetParams" in resp_json
            params = resp_json["datasetParams"]
            assert params['type'] == 'HDF5'
            assert params['path'] == path
            assert params['ds_path'] == "data"
