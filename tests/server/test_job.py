import json
import uuid

import pytest
import websockets

from utils import assert_msg

pytestmark = [pytest.mark.functional]


def _get_uuid_str():
    return str(uuid.uuid4())


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


async def create_connection(base_url, http_client):
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


async def create_default_dataset(default_raw, ws, http_client, base_url):
    ds_uuid = _get_uuid_str()
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

    return ds_uuid, ds_url


async def create_analysis(ws, http_client, base_url, ds_uuid, details=None):
    analysis_uuid = _get_uuid_str()
    analysis_url = "{}/api/analyses/{}/".format(base_url, analysis_uuid)
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
    async with http_client.put(analysis_url, json=analysis_data) as resp:
        print(await resp.text())
        assert resp.status == 200
        resp_json = await resp.json()
        assert resp_json['status'] == "ok"

    msg = json.loads(await ws.recv())
    assert_msg(msg, 'ANALYSIS_CREATED')
    assert msg['dataset'] == ds_uuid
    assert msg['analysis'] == analysis_uuid
    assert msg['details'] == details

    return analysis_uuid, analysis_url


async def create_update_compound_analysis(
    ws, http_client, base_url, ds_uuid, analyses, details=None, ca_uuid=None,
):
    if ca_uuid is None:
        ca_uuid = _get_uuid_str()
        creating = True
    else:
        creating = False
    ca_url = "{}/api/compoundAnalyses/{}/".format(base_url, ca_uuid)

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


async def create_job_for_analysis(ws, http_client, base_url, analysis_uuid):
    job_uuid = _get_uuid_str()
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

    return job_uuid, job_url


@pytest.mark.asyncio
async def test_run_job_1_sum(default_raw, base_url, http_client, server_port):
    await create_connection(base_url, http_client)

    print("checkpoint 1")

    # connect to ws endpoint:
    ws_url = "ws://127.0.0.1:{}/api/events/".format(server_port)
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid
        )

        await consume_task_results(ws, job_uuid)

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
    await create_connection(base_url, http_client)
    print("checkpoint 1")

    # connect to ws endpoint:
    ws_url = "ws://127.0.0.1:{}/api/events/".format(server_port)
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid
        )

        await consume_task_results(ws, job_uuid)

        # get rid of the dataset without cancelling the job:
        async with http_client.delete(ds_url) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DELETE_DATASET')


@pytest.mark.asyncio
async def test_cancel_unknown_job(default_raw, base_url, http_client, server_port):
    await create_connection(base_url, http_client)

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
    await create_connection(base_url, http_client)
    print("checkpoint 1")

    # connect to ws endpoint:
    ws_url = "ws://127.0.0.1:{}/api/events/".format(server_port)
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, details={
                "analysisType": "SUM_FRAMES",
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
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid
        )

        await consume_task_results(ws, job_uuid)

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
async def test_run_job_update_analysis_parameters(default_raw, base_url, http_client, server_port):
    await create_connection(base_url, http_client)
    print("checkpoint 1")

    # connect to ws endpoint:
    ws_url = "ws://127.0.0.1:{}/api/events/".format(server_port)
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid
        )

        await consume_task_results(ws, job_uuid)

        analysis_data_updated = {
            "dataset": ds_uuid,
            "details": {
                "analysisType": "SUM_FRAMES",
                "parameters": {
                    "roi": {
                        "shape": "disk",
                        "r": 1,
                        "cx": 1,
                        "cy": 1,
                    }
                }
            }
        }

        async with http_client.put(analysis_url, json=analysis_data_updated) as resp:
            print(await resp.text())
            assert resp.status == 200
            resp_json = await resp.json()
            assert resp_json['status'] == "ok"

        msg = json.loads(await ws.recv())
        assert_msg(msg, 'ANALYSIS_UPDATED')
        assert msg['analysis'] == analysis_uuid
        assert msg['details']['parameters'] == {
            "roi": {
                "shape": "disk",
                "r": 1,
                "cx": 1,
                "cy": 1,
            },
        }

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid
        )

        await consume_task_results(ws, job_uuid)

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
async def test_analysis_removal(default_raw, base_url, http_client, server_port):
    await create_connection(base_url, http_client)

    # connect to ws endpoint:
    ws_url = "ws://127.0.0.1:{}/api/events/".format(server_port)
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url
        )

        # compound analysis is first created without any analyses:
        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, analyses=[], details=None,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid
        )

        # compound analysis is updated with the newly created analysis:
        _, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, analyses=[], details={
                "mainType": "APPLY_RING_MASK",
                "analyses": [analysis_uuid]
            }, ca_uuid=ca_uuid
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid
        )

        await consume_task_results(ws, job_uuid)


@pytest.mark.asyncio
async def test_create_compound_analysis(default_raw, base_url, http_client, server_port):
    await create_connection(base_url, http_client)

    # connect to ws endpoint:
    ws_url = "ws://127.0.0.1:{}/api/events/".format(server_port)
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url
        )

        # compound analysis is first created without any analyses:
        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, analyses=[], details=None,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid
        )

        # compound analysis is updated with the newly created analysis:
        _, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, analyses=[], details={
                "mainType": "APPLY_RING_MASK",
                "analyses": [analysis_uuid]
            }, ca_uuid=ca_uuid
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid
        )

        await consume_task_results(ws, job_uuid)
