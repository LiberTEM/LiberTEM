import json

import pytest
import websockets

from libertem.io.dataset import register_dataset_cls, unregister_dataset_cls
from utils import assert_msg, MemoryDataSet

pytestmark = [pytest.mark.functional]


@pytest.fixture
def register_mem_ds():
    # FIXME: registering like this doesn't work, because it is not
    # communicated to worker processes
    register_dataset_cls("mem", MemoryDataSet)
    yield
    unregister_dataset_cls("mem")


def _get_ds_params():
    return {
        "dataset": {
            "params": {
                "type": "mem",
                "tileshape": [1, 6, 32, 64],
                "num_partitions": 4,
            }
        }
    }


@pytest.mark.xfail
@pytest.mark.asyncio
async def test_cancel_udf_job(base_url, http_client, server_port, register_mem_ds):
    conn_url = "{}/api/config/connection/".format(base_url)
    conn_details = {
        'connection': {
            'type': 'local',
            'numWorkers': 2,
        }
    }
    print("checkpoint 0")
    async with http_client.put(conn_url, json=conn_details) as response:
        assert response.status == 200
        assert (await response.json())['status'] == 'ok'

    print("checkpoint 1")

    # connect to ws endpoint:
    ws_url = "ws://127.0.0.1:{}/api/events/".format(server_port)
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid = "ae5d23bd-1f2a-4c57-bab2-dfc59a1219f3"
        ds_url = "{}/api/datasets/{}/".format(
            base_url, ds_uuid
        )
        ds_data = _get_ds_params()
        async with http_client.put(ds_url, json=ds_data) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'CREATE_DATASET')

        # same msg via ws:
        msg = json.loads(await ws.recv())
        assert_msg(msg, 'CREATE_DATASET')

        job_uuid = "229faa20-d146-46c1-af8c-32e303531322"
        job_url = "{}/api/jobs/{}/".format(base_url, job_uuid)
        job_data = {
            "job": {
                "dataset": ds_uuid,
                "analysis": {
                    "type": "SUM_FRAMES",
                    "parameters": {}
                }
            }
        }
        async with http_client.put(job_url, json=job_data) as resp:
            print(await resp.text())
            assert resp.status == 200
            resp_json = await resp.json()
            assert resp_json['status'] == "ok"

        msg = json.loads(await ws.recv())
        assert_msg(msg, 'JOB_STARTED')
        assert msg['job'] == job_uuid
        assert msg['details']['dataset'] == ds_uuid
        assert msg['details']['id'] == job_uuid

        num_followup = 0
        done = False
        while not done:
            msg = json.loads(await ws.recv())
            if msg['messageType'] == 'TASK_RESULT':
                assert_msg(msg, 'TASK_RESULT')
                assert msg['job'] == job_uuid
            elif msg['messageType'] == 'FINISH_JOB':
                done = True  # but we still need to check followup messages below
            elif msg['messageType'] == 'JOB_ERROR':
                raise Exception('JOB_ERROR: {}'.format(msg['msg']))
            else:
                raise Exception("invalid message type: {}".format(msg['messageType']))

            if 'followup' in msg:
                for i in range(msg['followup']['numMessages']):
                    msg = await ws.recv()
                    # followups should be PNG encoded:
                    assert msg[:8] == b'\x89\x50\x4E\x47\x0D\x0A\x1A\x0A'
                    num_followup += 1
        assert num_followup > 0

        # we are done with this job, clean up:
        async with http_client.delete(job_url) as resp:
            assert resp.status == 200
            assert_msg(await resp.json(), 'CANCEL_JOB')

        # also get rid of the dataset:
        async with http_client.delete(ds_url) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DELETE_DATASET')
