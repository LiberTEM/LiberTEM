import pytest

pytestmark = [pytest.mark.functional]


@pytest.mark.asyncio
async def test_local_cluster_details(base_url, http_client):
    url = "{}/api/config/connection/".format(base_url)
    conn_details = {
        'connection': {
            'type': 'local',
            'numWorkers': 2,
        }
    }

    async with http_client.put(url, json=conn_details) as response:
        assert response.status == 200

    url = "{}/api/config/cluster/".format(base_url)
    async with http_client.get(url) as response:
        assert response.status == 200
        details = await response.json()
        assert details == {
            "status": "ok",
            "messageType": "CLUSTER_DETAILS",
            "details": [{
                "host": "default",
                "cpu": 2,
                "cuda": 0,
                "service": 1
            }]
        }
