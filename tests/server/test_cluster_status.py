import pytest
import random

pytestmark = [pytest.mark.web_api]


@pytest.mark.dist
@pytest.mark.asyncio
async def test_tcp_cluster_details(dist_ctx, base_url, http_client, default_token):

    worker_set = dist_ctx.executor.get_available_workers()
    host1, host2 = sorted(worker_set.hosts())
    url = f"{base_url}/api/config/connection/?token={default_token}"

    conn_details = {
        'connection': {
            'type': 'TCP',
            'address': 'tcp://scheduler:8786',
        }
    }
    async with http_client.put(url, json=conn_details) as response:
        assert response.status == 200

    url = f"{base_url}/api/config/cluster/?token={default_token}"
    async with http_client.get(url) as response:
        assert response.status == 200
        details = await response.json()
        assert details == {
            "status": "ok",
            "messageType": "CLUSTER_DETAILS",
            "details": [
                {
                    'cpu': 2,
                    'cuda': 0,
                    'host': host1,
                    'service': 1
                },
                {
                    'cpu': 2,
                    'cuda': 0,
                    'host': host2,
                    'service': 1
                }
            ]
        }


@pytest.mark.asyncio
async def test_local_cluster_details(shared_state, base_url, http_client, default_token):
    num_workers = random.randint(1, 4)
    url = f"{base_url}/api/config/connection/?token={default_token}"
    conn_details = {
        'connection': {
            'type': 'local',
            'numWorkers': num_workers,
        }
    }

    async with http_client.put(url, json=conn_details) as response:
        assert response.status == 200

    executor = await shared_state.executor_state.get_executor()
    worker_set = await executor.get_available_workers()
    host = worker_set.hosts().pop()

    url = f"{base_url}/api/config/cluster/?token={default_token}"
    async with http_client.get(url) as response:
        assert response.status == 200
        details = await response.json()
        assert details == {
            "status": "ok",
            "messageType": "CLUSTER_DETAILS",
            "details": [{
                "host": host,
                "cpu": num_workers,
                "cuda": 0,
                "service": 1
            }]
        }
