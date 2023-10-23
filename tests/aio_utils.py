import json
import uuid
import urllib.parse as urlparse
from urllib.parse import urlencode

from utils import assert_msg


def _get_raw_params(path):
    return {
        "dataset": {
            "params": {
                "type": "raw",
                "path": path,
                "name": "name.raw",
                "dtype": "float32",
                "sig_shape": [128, 128],
                "enable_direct": False,
                "nav_shape": [16, 16],
                "sync_offset": 0
            }
        }
    }


def _get_uuid_str():
    return str(uuid.uuid4())


def add_token(url, token):
    if token is None:
        return url
    url_parts = list(urlparse.urlparse(url))
    query = dict(urlparse.parse_qsl(url_parts[4]))
    query['token'] = token
    url_parts[4] = urlencode(query)
    return urlparse.urlunparse(url_parts)


async def create_connection(base_url, http_client, scheduler_url=None, token=None):
    conn_url = f"{base_url}/api/config/connection/"
    conn_url = add_token(conn_url, token)
    if scheduler_url is None:
        conn_details = {
            'connection': {
                'type': 'local',
                'numWorkers': 2,
                'cudas': {},
            }
        }
    else:
        conn_details = {
            'connection': {
                'type': 'tcp',
                'address': scheduler_url,
            }
        }
    print("checkpoint 0")
    async with http_client.put(conn_url, json=conn_details) as response:
        assert response.status == 200
        response_json = await response.json()
        assert response_json['status'] == 'ok'


async def consume_task_results(ws, job_uuid):
    num_followup = 0
    done = False
    while not done:
        msg = json.loads(await ws.recv())
        if msg['messageType'] == 'TASK_RESULT':
            assert_msg(msg, 'TASK_RESULT')
            assert msg['job'] == job_uuid
        elif msg['messageType'] == 'FINISH_JOB':
            done = True  # but we still need to check followup messages below
        elif msg['messageType'] == 'JOB_PROGRESS':
            assert msg['job'] == job_uuid
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


async def create_default_dataset(default_raw, ws, http_client, base_url, token=None):
    ds_uuid = _get_uuid_str()
    ds_url = "{}/api/datasets/{}/".format(
        base_url, ds_uuid
    )
    ds_url = add_token(ds_url, token)
    ds_data = _get_raw_params(default_raw._path)

    print("creating dataset: dispatching PUT request")

    async with http_client.put(ds_url, json=ds_data) as resp:
        assert resp.status == 200
        resp_json = await resp.json()
        assert_msg(resp_json, 'CREATE_DATASET')

    print("PUT request done, waiting for websocket response")

    # same msg via ws:
    msg = json.loads(await ws.recv())
    assert_msg(msg, 'CREATE_DATASET')

    return ds_uuid, ds_url


async def create_analysis(ws, http_client, base_url, ds_uuid, ca_uuid, details=None, token=None):
    analysis_uuid = _get_uuid_str()
    analysis_url = "{}/api/compoundAnalyses/{}/analyses/{}/".format(
        base_url, ca_uuid, analysis_uuid
    )
    analysis_url = add_token(analysis_url, token)
    if details is None:
        details = {
            "analysisType": "SUM_FRAMES",
            "parameters": {}
        }
    else:
        assert "analysisType" in details
        assert "parameters" in details
    analysis_data = {
        "dataset": ds_uuid,
        "details": details,
    }
    print("before PUT")
    async with http_client.put(analysis_url, json=analysis_data) as resp:
        print(await resp.text())
        assert resp.status == 200
        resp_json = await resp.json()
        assert resp_json['status'] == "ok"
    print("after PUT")

    msg = json.loads(await ws.recv())
    assert_msg(msg, 'ANALYSIS_CREATED')
    assert msg['dataset'] == ds_uuid
    assert msg['analysis'] == analysis_uuid
    assert msg['details'] == details

    return analysis_uuid, analysis_url


async def update_analysis(
    ws, http_client, base_url, ds_uuid, ca_uuid, analysis_uuid, details, token=None,
):
    analysis_url = "{}/api/compoundAnalyses/{}/analyses/{}/".format(
        base_url, ca_uuid, analysis_uuid
    )
    analysis_url = add_token(analysis_url, token)
    assert "analysisType" in details
    assert "parameters" in details
    analysis_data = {
        "dataset": ds_uuid,
        "details": details,
    }
    async with http_client.put(analysis_url, json=analysis_data) as resp:
        print(await resp.text())
        assert resp.status == 200
        resp_json = await resp.json()
        assert resp_json['status'] == "ok"

    msg = json.loads(await ws.recv())
    assert_msg(msg, 'ANALYSIS_UPDATED')
    assert msg['dataset'] == ds_uuid
    assert msg['analysis'] == analysis_uuid
    assert msg['details'] == details


async def create_update_compound_analysis(
    ws, http_client, base_url, ds_uuid, details=None, ca_uuid=None, token=None,
):
    if ca_uuid is None:
        ca_uuid = _get_uuid_str()
        creating = True
    else:
        creating = False
    ca_url = f"{base_url}/api/compoundAnalyses/{ca_uuid}/"
    ca_url = add_token(ca_url, token)

    if details is None:
        details = {
            "mainType": "APPLY_RING_MASK",
            "analyses": [],
        }
    else:
        assert "analyses" in details
        assert "mainType" in details

    ca_data = {
        "dataset": ds_uuid,
        "details": details,
    }

    print("dispatching PUT request...")

    async with http_client.put(ca_url, json=ca_data) as resp:
        print(await resp.text())

        assert resp.status == 200
        resp_json = await resp.json()
        assert resp_json['status'] == "ok"

    msg = json.loads(await ws.recv())
    if creating:
        assert_msg(msg, 'COMPOUND_ANALYSIS_CREATED')
    else:
        assert_msg(msg, 'COMPOUND_ANALYSIS_UPDATED')
    assert msg['dataset'] == ds_uuid
    assert msg['compoundAnalysis'] == ca_uuid
    assert msg['details'] == details

    return ca_uuid, ca_url


async def create_job_for_analysis(ws, http_client, base_url, analysis_uuid, token=None):
    job_uuid = _get_uuid_str()
    job_url = f"{base_url}/api/jobs/{job_uuid}/"
    job_data = {
        "job": {
            "analysis": analysis_uuid,
        }
    }
    job_url = add_token(job_url, token)
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

    return job_uuid, job_url


async def call_ca_rpc(http_client, base_url, ca_uuid, rpc_name, token=None):
    rpc_url = f"{base_url}/api/compoundAnalyses/{ca_uuid}/rpc/{rpc_name}/"
    rpc_url = add_token(rpc_url, token)

    print(f"call CA before PUT {rpc_url}")

    async with http_client.put(rpc_url) as resp:
        print("call CA getting the text response:")
        print(await resp.text())
        resp_json = await resp.json()
        return resp_json
