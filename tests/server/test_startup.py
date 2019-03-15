import json

import pytest
import websockets


@pytest.mark.asyncio
async def test_start_server(base_url, http_client):
    """
    start the server and HTTP GET '/', which should
    return a 200 status code
    """
    url = base_url
    async with http_client.get(url) as response:
        assert response.status == 200
        print(await response.text())


@pytest.mark.asyncio
async def test_get_config(base_url, http_client):
    url = "{}/api/config/".format(base_url)
    async with http_client.get(url) as response:
        assert response.status == 200
        config = await response.json()
        assert set(config.keys()) == set(["status", "messageType", "config"])
        assert set(config['config'].keys()) == set([
            "version", "revision", "localCores", "cwd", "separator"
        ])


@pytest.mark.asyncio
async def test_conn_is_disconnected(base_url, http_client):
    url = "{}/api/config/connection/".format(base_url)
    async with http_client.get(url) as response:
        assert response.status == 200
        conn = await response.json()
        print(conn)
        assert conn['status'] == 'disconnected'
        assert conn['connection'] == {}


@pytest.mark.asyncio
async def test_conn_connect_local(base_url, http_client):
    url = "{}/api/config/connection/".format(base_url)
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
async def test_initial_state_empty(default_raw, base_url, http_client, server_port):
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
        assert initial_msg['messageType'] == "INITIAL_STATE"
        assert initial_msg['status'] == "ok"
        assert initial_msg['datasets'] == []
        assert initial_msg['jobs'] == []
