import json

import pytest
import websockets

from utils import assert_msg
from aio_utils import create_connection

from libertem.executor.inline import InlineJobExecutor
from libertem.executor.base import AsyncAdapter

pytestmark = [pytest.mark.web_api]


def _get_raw_params(path):
    return {
        "dataset": {
            "params": {
                "type": "RAW",
                "path": path,
                "dtype": "float32",
                "sig_shape": [128, 128],
                "enable_direct": False,
                "nav_shape": [16, 16],
                "sync_offset": 0
            }
        }
    }


@pytest.mark.asyncio
async def test_load_raw_success(
    default_raw, base_url, http_client, local_cluster_url, default_token
):
    await create_connection(base_url, http_client, local_cluster_url, default_token)
    raw_path = default_raw._path

    uuid = "ae5d23bd-1f2a-4c57-bab2-dfc59a1219f3"
    ds_url = "{}/api/datasets/{}/?token={}".format(
        base_url, uuid, default_token,
    )
    ds_data = _get_raw_params(raw_path)
    async with http_client.put(ds_url, json=ds_data) as resp:
        assert resp.status == 200
        resp_json = await resp.json()
        assert_msg(resp_json, 'CREATE_DATASET')
        for k in ds_data['dataset']['params']:
            assert ds_data['dataset']['params'][k] == resp_json['details']['params'][k]


@pytest.mark.asyncio
async def test_load_raw_with_backends(
    default_raw, base_url, http_client, local_cluster_url, default_token
):
    await create_connection(base_url, http_client, local_cluster_url, default_token)
    raw_path = default_raw._path

    uuid = "ae5d23bd-1f2a-4c57-bab2-dfc59a1219f3"
    ds_url = "{}/api/datasets/{}/?token={}".format(
        base_url, uuid, default_token,
    )
    ds_data = _get_raw_params(raw_path)
    for backend in ["mmap", "buffered"]:
        ds_data["dataset"]["params"]["io_backend"] = backend
        async with http_client.put(ds_url, json=ds_data) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'CREATE_DATASET')
            for k in ds_data['dataset']['params']:
                assert ds_data['dataset']['params'][k] == resp_json['details']['params'][k]
            assert ds_data["dataset"]["params"]["io_backend"] == backend
        async with http_client.delete(ds_url) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DELETE_DATASET')


@pytest.mark.asyncio
async def test_load_raw_fail(base_url, http_client, local_cluster_url, default_token):
    await create_connection(base_url, http_client, local_cluster_url, default_token)

    uuid = "ae5d23bd-1f2a-4c57-bab2-dfc59a1219f3"
    ds_url = "{}/api/datasets/{}/?token={}".format(
        base_url, uuid, default_token,
    )
    ds_data = _get_raw_params("/does/not/exist/")
    async with http_client.put(ds_url, json=ds_data) as resp:
        assert resp.status == 200
        resp_json = await resp.json()
        assert_msg(resp_json, 'CREATE_DATASET_ERROR', status='error')
        assert resp_json['dataset'] == uuid
        assert (
            "No such file or directory" in resp_json['msg']
            or "The system cannot find the path specified" in resp_json['msg']
        )


@pytest.mark.asyncio
async def test_dataset_delete(
    default_raw, base_url, http_client, server_port, local_cluster_url, default_token,
):
    await create_connection(base_url, http_client, local_cluster_url, default_token)
    raw_path = default_raw._path

    uuid = "ae5d23bd-1f2a-4c57-bab2-dfc59a1219f3"
    ds_url = "{}/api/datasets/{}/?token={}".format(
        base_url, uuid, default_token,
    )
    ds_data = _get_raw_params(raw_path)

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/?token={default_token}"
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
async def test_initial_state_after_reconnect(
    default_raw, base_url, http_client, server_port, local_cluster_url, default_token,
):
    await create_connection(base_url, http_client, local_cluster_url, default_token)
    raw_path = default_raw._path

    uuid = "ae5d23bd-1f2a-4c57-bab2-dfc59a1219f3"
    ds_url = "{}/api/datasets/{}/?token={}".format(
        base_url, uuid, default_token,
    )
    ds_data = _get_raw_params(raw_path)

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/?token={default_token}"
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
            'sig_shape': [128, 128],
            "enable_direct": False,
            'dtype': 'float32',
            'path': raw_path,
            'nav_shape': [16, 16],
            'shape': [16, 16, 128, 128],
            'type': 'RAW',
            'sync_offset': 0
        }
        assert len(initial_msg["datasets"][0]["diagnostics"]) == 7


@pytest.mark.asyncio
async def test_prime_cache(
    shared_state, default_raw, base_url, http_client, server_port, local_cluster_url,
    default_token,
):
    # first, connect to get the state
    await create_connection(base_url, http_client, local_cluster_url, default_token)

    executor = InlineJobExecutor()

    pool = AsyncAdapter.make_pool()
    executor = AsyncAdapter(wrapped=executor, pool=pool)
    conn_details = {
        'connection': {
            'type': 'local',
            'numWorkers': 1,
            'cudas': [],
        }
    }
    await shared_state.executor_state.set_executor(executor, conn_details)

    raw_path = default_raw._path

    uuid = "ae5d23bd-1f2a-4c57-bab2-dfc59a1219f3"
    ds_url = "{}/api/datasets/{}/?token={}".format(
        base_url, uuid, default_token,
    )
    ds_data = _get_raw_params(raw_path)

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/?token={default_token}"
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
            'sig_shape': [128, 128],
            "enable_direct": False,
            'dtype': 'float32',
            'path': raw_path,
            'nav_shape': [16, 16],
            'shape': [16, 16, 128, 128],
            'type': 'RAW',
            'sync_offset': 0
        }
        assert len(initial_msg["datasets"][0]["diagnostics"]) == 7
