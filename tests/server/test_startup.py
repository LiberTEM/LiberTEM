import pytest


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
