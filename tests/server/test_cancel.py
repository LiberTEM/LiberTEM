import json
import asyncio

import pytest
import websockets

from utils import assert_msg


pytestmark = [pytest.mark.functional]


def _get_ds_params():
    return {
        "dataset": {
            "params": {
                "type": "memory",
                "tileshape": [7, 32, 32],
                "datashape": [256, 32, 32],
                "tiledelay": 0.1,
                "num_partitions": 16,
            }
        }
    }


@pytest.mark.asyncio
async def test_cancel_udf_job(base_url, http_client, server_port, shared_data):
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

        await asyncio.sleep(0)  # for debugging, set to >0

        async with http_client.delete(job_url) as resp:
            assert resp.status == 200
            assert_msg(await resp.json(), 'CANCEL_JOB')

        # wait for CANCEL_JOB message:
        done = False
        num_seen = 0
        while not done:
            msg = json.loads(await ws.recv())
            num_seen += 1
            if msg['messageType'] == 'TASK_RESULT':
                assert_msg(msg, 'TASK_RESULT')
                assert msg['job'] == job_uuid
            elif msg['messageType'] == 'CANCEL_JOB':
                # this is the confirmation sent from the DELETE method handler:
                assert_msg(msg, 'CANCEL_JOB')
                assert msg['job'] == job_uuid
                done = True
            else:
                raise Exception("invalid message type: {}".format(msg['messageType']))
            if 'followup' in msg:
                for i in range(msg['followup']['numMessages']):
                    # drain binary messages:
                    msg = await ws.recv()

        assert num_seen < 4
        assert job_uuid not in shared_data.jobs

        # now we drain messages from websocket and look for CANCEL_JOB_DONE msg:
        done = False
        num_seen = 0
        types_seen = []
        msgs = []
        while not done:
            msg = json.loads(await ws.recv())
            print(msg)
            num_seen += 1
            types_seen.append(msg['messageType'])
            msgs.append(msg)
            if msg['messageType'] == 'TASK_RESULT':
                assert_msg(msg, 'TASK_RESULT')
                assert msg['job'] == job_uuid
            elif msg['messageType'] == 'CANCEL_JOB_DONE':
                assert msg['job'] == job_uuid
                done = True
            elif msg['messageType'] == 'FINISH_JOB':
                done = True  # we should not get this one...
            elif msg['messageType'] == 'JOB_ERROR':
                raise Exception('JOB_ERROR: {}'.format(msg['msg']))
            else:
                raise Exception("invalid message type: {}".format(msg['messageType']))
            if 'followup' in msg:
                for i in range(msg['followup']['numMessages']):
                    # drain binary messages:
                    msg = await ws.recv()
        assert set(types_seen) == {"CANCEL_JOB_DONE"}
        assert num_seen < 4

        # get rid of the dataset:
        async with http_client.delete(ds_url) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DELETE_DATASET')
