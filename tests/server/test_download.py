import io
import json

import h5py
import pytest
import websockets
import numpy as np

from utils import assert_msg


def _get_raw_params(path):
    return {
        "dataset": {
            "params": {
                "type": "raw",
                "path": path,
                "dtype": "float32",
                "detector_size": [128, 128],
                "enable_direct": False,
                "scan_size": [16, 16]
            }
        }
    }


@pytest.mark.asyncio
async def test_download_1(default_raw, base_url, http_client, server_port):
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

        analysis_uuid = "229faa20-d146-46c1-af8c-32e303531322"
        analysis_url = "{}/api/analyses/{}/".format(base_url, analysis_uuid)
        analysis_data = {
            "dataset": ds_uuid,
            "analysis": {
                "type": "SUM_FRAMES",
                "parameters": {}
            }
        }
        async with http_client.put(analysis_url, json=analysis_data) as resp:
            print(await resp.text())
            assert resp.status == 200
            resp_json = await resp.json()
            assert resp_json['status'] == "ok"

        msg = json.loads(await ws.recv())
        assert_msg(msg, 'ANALYSIS_CREATED')
        assert msg['dataset'] == ds_uuid
        assert msg['analysis'] == analysis_uuid
        assert msg['details']['parameters'] == {}

        job_uuid = "229faa20-d146-46c1-af8c-32e303531322"
        job_url = "{}/api/jobs/{}/".format(base_url, job_uuid)
        job_data = {
            "job": {
                "analysis": analysis_uuid,
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
        assert msg['analysis'] == analysis_uuid
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

        download_url = "{}/api/analyses/{}/download/?format=h5".format(base_url, analysis_uuid)
        async with http_client.get(download_url) as resp:
            raw_data = await resp.read()
            bio = io.BytesIO(raw_data)
            with h5py.File(bio, "r") as f:
                assert 'intensity' in f.keys()
                data = np.array(f['intensity'])
                default_raw_data = np.memmap(
                    filename=default_raw._path,
                    dtype=default_raw.dtype,
                    mode='r',
                    shape=tuple(default_raw.shape),
                )
                assert np.allclose(
                    default_raw_data.sum(axis=(0, 1)),
                    data
                )

        # we are done with this analysis, clean up:
        async with http_client.delete(analysis_url) as resp:
            assert resp.status == 200
            assert_msg(await resp.json(), 'ANALYSIS_REMOVED')

        # we are done with this job, clean up:
        async with http_client.delete(job_url) as resp:
            assert resp.status == 200
            assert_msg(await resp.json(), 'CANCEL_JOB')

        # also get rid of the dataset:
        async with http_client.delete(ds_url) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DELETE_DATASET')
