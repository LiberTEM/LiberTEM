import json

import pytest
import websockets

from utils import assert_msg

pytestmark = [pytest.mark.functional]


def _get_raw_params(path):
    return {
        "dataset": {
            "params": {
                "type": "raw",
                "path": path,
                "dtype": "float32",
                "detector_size": [128, 128],
                "enable_direct": False,
                "tileshape": [1, 1, 128, 128],
                "scan_size": [16, 16]
            }
        }
    }


@pytest.mark.asyncio
async def test_run_job_1_sum(default_raw, base_url, http_client, server_port):
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
        ds_data = _get_raw_params(default_raw._path)
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


@pytest.mark.asyncio
async def test_run_job_delete_ds(default_raw, base_url, http_client, server_port):
    """
    main difference to test above: we just close the dataset without
    removing the job first. this tests another code path in `remove_dataset`
    """
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
        ds_data = _get_raw_params(default_raw._path)
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

        # get rid of the dataset without cancelling the job:
        async with http_client.delete(ds_url) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DELETE_DATASET')


@pytest.mark.asyncio
async def test_cancel_unknown_job(default_raw, base_url, http_client, server_port):
    conn_url = "{}/api/config/connection/".format(base_url)
    conn_details = {
        'connection': {
            'type': 'local',
            'numWorkers': 2,
        }
    }
    async with http_client.put(conn_url, json=conn_details) as response:
        assert response.status == 200
        assert (await response.json())['status'] == 'ok'

    # connect to ws endpoint:
    ws_url = "ws://127.0.0.1:{}/api/events/".format(server_port)
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        job_uuid = "un-kn-ow-n"
        job_url = "{}/api/jobs/{}/".format(base_url, job_uuid)

        # try to cancel unknown job:
        async with http_client.delete(job_url) as resp:
            assert resp.status == 200
            assert_msg(await resp.json(), 'CANCEL_JOB_FAILED', status='error')


@pytest.mark.asyncio
async def test_run_with_all_zeros_roi(default_raw, base_url, http_client, server_port):
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
        ds_data = _get_raw_params(default_raw._path)
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
                    "parameters": {
                        "roi": {
                            "shape": "disk",
                            "r": 0,
                            # setting the center outside of the frame
                            # means our ROI will be all-zero
                            "cx": -1,
                            "cy": -1,
                        }
                    }
                }
            }
        }

        print("creating job", job_data)
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
