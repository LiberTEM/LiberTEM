import time
import json
import sys
import asyncio
import signal

import pytest
import websockets

from utils import assert_msg
from aio_utils import (
    create_connection, create_analysis, create_update_compound_analysis,
)

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
async def test_start_server(base_url, http_client, default_token):
    """
    start the server and HTTP GET '/', which should
    return a 200 status code
    """
    url = f"{base_url}?token={default_token}"
    async with http_client.get(url) as response:
        assert response.status == 200
        print(await response.text())


@pytest.mark.asyncio
async def test_get_config(base_url, http_client, default_token):
    url = f"{base_url}/api/config/?token={default_token}"
    async with http_client.get(url) as response:
        assert response.status == 200
        config = await response.json()
        assert set(config.keys()) == {"status", "messageType", "config"}
        assert set(config['config'].keys()) == {
            "version", "revision", "localCores", "cwd",
            "separator", "resultFileFormats", "devices",
            "datasetTypes",
        }


@pytest.mark.asyncio
async def test_conn_is_disconnected(base_url, http_client, default_token):
    url = f"{base_url}/api/config/connection/?token={default_token}"
    async with http_client.get(url) as response:
        assert response.status == 200
        conn = await response.json()
        print(conn)
        assert conn['status'] == 'disconnected'
        assert conn['connection'] == {}


@pytest.mark.asyncio
async def test_conn_connect_local(base_url, http_client, default_token):
    url = f"{base_url}/api/config/connection/?token={default_token}"
    conn_details = {
        'connection': {
            'type': 'local',
            'numWorkers': 2,
        }
    }
    async with http_client.put(url, json=conn_details) as response:
        assert response.status == 200
        conn_resp = await response.json()
        assert conn_resp == {
            "status": "ok",
            "connection": {
                "type": "local",
                "numWorkers": 2,
            }
        }

    async with http_client.get(url) as response:
        assert response.status == 200
        conn = await response.json()
        assert conn == {
            "status": "ok",
            "connection": {
                "type": "local",
                "numWorkers": 2,
            }
        }


@pytest.mark.asyncio
async def test_cluster_create_error(base_url, http_client, default_token):
    url = f"{base_url}/api/config/connection/?token={default_token}"
    conn_details = {
        'connection': {
            'type': 'local',
            'numWorkers': 'foo',
        }
    }
    async with http_client.put(url, json=conn_details) as response:
        conn_resp = await response.json()
        assert response.status == 500
        assert_msg(conn_resp, 'CLUSTER_CONN_ERROR', status='error')


@pytest.mark.asyncio
async def test_cluster_connect_error(base_url, http_client, default_token):
    url = f"{base_url}/api/config/connection/?token={default_token}"
    conn_details = {
        'connection': {
            'type': 'TCP',
            'address': 'tcp://invalid',
        }
    }
    async with http_client.put(url, json=conn_details) as response:
        conn_resp = await response.json()
        assert response.status == 500
        assert_msg(conn_resp, 'CLUSTER_CONN_ERROR', status='error')


@pytest.mark.asyncio
async def test_initial_state_empty(
    default_raw, base_url, http_client, server_port, local_cluster_url, default_token,
):
    conn_url = f"{base_url}/api/config/connection/?token={default_token}"
    conn_details = {
        'connection': {
            'type': 'tcp',
            'address': local_cluster_url,
        }
    }
    async with http_client.put(conn_url, json=conn_details) as response:
        assert response.status == 200

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/?token={default_token}"
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert initial_msg['messageType'] == "INITIAL_STATE"
        assert initial_msg['status'] == "ok"
        assert initial_msg['datasets'] == []
        assert initial_msg['jobs'] == []
        assert initial_msg['analyses'] == []


@pytest.mark.asyncio
async def test_initial_state_w_existing_ds(
    default_raw, base_url, http_client, server_port, local_cluster_url, default_token,
):
    conn_url = f"{base_url}/api/config/connection/?token={default_token}"
    conn_details = {
        'connection': {
            'type': 'tcp',
            'address': local_cluster_url,
        }
    }
    async with http_client.put(conn_url, json=conn_details) as response:
        assert response.status == 200

    # first connect has empty list of datasets:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/?token={default_token}"
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert initial_msg['messageType'] == "INITIAL_STATE"
        assert initial_msg['status'] == "ok"
        assert initial_msg['datasets'] == []

    raw_path = default_raw._path

    ds_uuid = "ae5d23bd-1f2a-4c57-bab2-dfc59a1219f3"
    ds_url = "{}/api/datasets/{}/?token={}".format(
        base_url, ds_uuid, default_token,
    )
    ds_params = _get_raw_params(raw_path)
    async with http_client.put(ds_url, json=ds_params) as resp:
        assert resp.status == 200
        resp_json = await resp.json()
        assert_msg(resp_json, 'CREATE_DATASET')
        for k in ds_params['dataset']['params']:
            assert ds_params['dataset']['params'][k] == resp_json['details']['params'][k]

    # second connect has one dataset:
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert initial_msg['messageType'] == "INITIAL_STATE"
        assert initial_msg['status'] == "ok"
        assert len(initial_msg['datasets']) == 1
        assert initial_msg['datasets'][0]["id"] == ds_uuid
        assert initial_msg['datasets'][0]["params"] == {
            "type": "RAW",
            "path": raw_path,
            "dtype": "float32",
            "sig_shape": [128, 128],
            "enable_direct": False,
            "nav_shape": [16, 16],
            "shape": [16, 16, 128, 128],
            "sync_offset": 0
        }


@pytest.mark.asyncio
async def test_initial_state_analyses(
    default_raw, base_url, http_client, server_port, local_cluster_url, default_token,
):
    await create_connection(base_url, http_client, local_cluster_url, default_token)

    # first connect has empty list of datasets:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/?token={default_token}"
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert initial_msg['messageType'] == "INITIAL_STATE"
        assert initial_msg['status'] == "ok"
        assert initial_msg['datasets'] == []

    raw_path = default_raw._path

    ds_uuid = "ae5d23bd-1f2a-4c57-bab2-dfc59a1219f3"
    ds_url = "{}/api/datasets/{}/?token={}".format(
        base_url, ds_uuid, default_token,
    )
    ds_params = _get_raw_params(raw_path)
    async with http_client.put(ds_url, json=ds_params) as resp:
        assert resp.status == 200
        resp_json = await resp.json()
        assert_msg(resp_json, 'CREATE_DATASET')
        for k in ds_params['dataset']['params']:
            assert ds_params['dataset']['params'][k] == resp_json['details']['params'][k]

        async with websockets.connect(ws_url) as ws:
            initial_msg = json.loads(await ws.recv())
            assert initial_msg['messageType'] == "INITIAL_STATE"

            ca_uuid, ca_url = await create_update_compound_analysis(
                ws, http_client, base_url, ds_uuid, token=default_token,
            )

            analysis_uuid, analysis_url = await create_analysis(
                ws, http_client, base_url, ds_uuid, ca_uuid,
                details={
                    "analysisType": "SUM_FRAMES",
                    "parameters": {
                        "roi": {
                            "shape": "disk",
                            "r": 1,
                            "cx": 1,
                            "cy": 1,
                        }
                    }
                }, token=default_token,
            )

    # second connect has one dataset and one analysis:
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert initial_msg['messageType'] == "INITIAL_STATE"
        assert initial_msg['status'] == "ok"
        assert len(initial_msg['datasets']) == 1
        assert len(initial_msg['analyses']) == 1
        assert initial_msg["analyses"][0]["details"]["parameters"] == {
            "roi": {
                "shape": "disk",
                "r": 1,
                "cx": 1,
                "cy": 1,
            },
        }


@pytest.mark.asyncio
async def test_libertem_server_cli_startup():
    # make sure we can start `libertem-server` and stop it again using ctrl+c
    # this is kind of a smoke test, which should cover the main cli functions.
    p = await asyncio.create_subprocess_exec(
        sys.executable, '-m', 'libertem.web.cli', '--no-browser',
        stderr=asyncio.subprocess.PIPE,
    )
    # total deadline, basically how long it takes to import all the dependencies
    # and start the web API
    # (no executor startup is done here)
    deadline = time.monotonic() + 15
    while True:
        if time.monotonic() > deadline:
            assert False, 'timeout'
        line = await asyncio.wait_for(p.stderr.readline(), 5)
        if not line:  # EOF
            assert False, 'subprocess is dead'
        line = line.decode("utf8")
        print('Line:', line, end='')
        if 'LiberTEM listening on' in line:
            break

    async def _debug():
        while True:
            line = await asyncio.wait_for(p.stderr.readline(), 5)
            if not line:  # EOF
                return
            line = line.decode("utf8")
            print('Line@_debug:', line, end='')

    asyncio.ensure_future(_debug())

    try:
        # now, let's kill the subprocess:
        # ctrl+s twice should do the job:
        p.send_signal(signal.SIGINT)
        await asyncio.sleep(0.5)
        if p.returncode is None:
            p.send_signal(signal.SIGINT)

        # wait for the process to stop, but max. 1 second:
        ret = await asyncio.wait_for(p.wait(), 1)
        assert ret == 0
    except Exception:
        if p.returncode is None:
            p.terminate()
            await asyncio.sleep(0.2)
        if p.returncode is None:
            p.kill()
        raise
