import json
import asyncio

import pytest
import websockets

from utils import assert_msg
from aio_utils import (
    create_connection, create_analysis,
    create_update_compound_analysis, create_job_for_analysis,
)


pytestmark = [pytest.mark.web_api]


def _get_ds_params():
    return {
        "dataset": {
            "params": {
                "type": "memory",
                "tileshape": [7, 32, 32],
                "datashape": [256, 32, 32],
                "tiledelay": 0.1,
                "num_partitions": 16,
                "sync_offset": 0,
            }
        }
    }


@pytest.mark.asyncio
async def test_cancel_udf_job(
    base_url, default_raw, http_client, server_port, shared_state, local_cluster_url, default_token,
):
    await create_connection(base_url, http_client, local_cluster_url, default_token)

    print("checkpoint 1")

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/?token={default_token}"
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid = "ae5d23bd-1f2a-4c57-bab2-dfc59a1219f3"
        ds_url = "{}/api/datasets/{}/?token={}".format(
            base_url, ds_uuid, default_token,
        )

        ds_data = _get_ds_params()
        async with http_client.put(ds_url, json=ds_data) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'CREATE_DATASET')

        # same msg via ws:
        msg = json.loads(await ws.recv())
        assert_msg(msg, 'CREATE_DATASET')

        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, token=default_token,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, ca_uuid, token=default_token,
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid, token=default_token,
        )

        await asyncio.sleep(0)  # for debugging, set to >0

        async with http_client.delete(job_url) as resp:
            assert resp.status == 200
            assert_msg(await resp.json(), 'CANCEL_JOB')

        # wait for CANCEL_JOB message:
        done = False
        num_seen = 0
        while not done:
            msg = json.loads(await ws.recv())
            if msg['messageType'] != "JOB_PROGRESS":
                num_seen += 1
            if msg['messageType'] == 'TASK_RESULT':
                assert_msg(msg, 'TASK_RESULT')
                assert msg['job'] == job_uuid
            elif msg['messageType'] == 'JOB_PROGRESS':
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
        assert job_uuid not in shared_state.job_state.jobs

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
            elif msg['messageType'] == 'JOB_PROGRESS':
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
        for mtype in types_seen:
            assert mtype in {"CANCEL_JOB_DONE", "JOB_PROGRESS"}
        msgs_without_progress = [
            msg
            for msg in msgs
            if msg['messageType'] != "JOB_PROGRESS"
        ]
        assert len(msgs_without_progress) < 4

        # get rid of the dataset:
        async with http_client.delete(ds_url) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DELETE_DATASET')
