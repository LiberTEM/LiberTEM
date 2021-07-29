import json

import pytest
import websockets

from utils import assert_msg
from aio_utils import (
    create_connection, consume_task_results, create_default_dataset, create_analysis,
    create_update_compound_analysis, create_job_for_analysis,
)


pytestmark = [pytest.mark.functional]


@pytest.mark.asyncio
async def test_run_job_1_sum(default_raw, base_url, http_client, server_port, local_cluster_url):
    await create_connection(base_url, http_client, scheduler_url=local_cluster_url)

    print("checkpoint 1")

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/"
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url
        )

        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, ca_uuid,
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
async def test_run_job_delete_ds(
    default_raw, base_url, http_client, server_port, local_cluster_url
):
    """
    main difference to test above: we just close the dataset without
    removing the job first. this tests another code path in `remove_dataset`
    """
    await create_connection(base_url, http_client, scheduler_url=local_cluster_url)
    print("checkpoint 1")

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/"
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url
        )

        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, ca_uuid,
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
async def test_cancel_unknown_job(
    default_raw, base_url, http_client, server_port, local_cluster_url
):
    await create_connection(base_url, http_client, scheduler_url=local_cluster_url)

    ws_url = f"ws://127.0.0.1:{server_port}/api/events/"
    async with websockets.connect(ws_url) as ws:
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        job_uuid = "un-kn-ow-n"
        job_url = f"{base_url}/api/jobs/{job_uuid}/"

        # try to cancel unknown job:
        async with http_client.delete(job_url) as resp:
            assert resp.status == 200
            assert_msg(await resp.json(), 'CANCEL_JOB_FAILED', status='error')


@pytest.mark.asyncio
async def test_run_with_all_zeros_roi(
    default_raw, base_url, http_client, server_port, local_cluster_url
):
    await create_connection(base_url, http_client, scheduler_url=local_cluster_url)
    print("checkpoint 1")

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/"
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url
        )

        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, ca_uuid, details={
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
async def test_run_job_update_analysis_parameters(
    default_raw, base_url, http_client, server_port, local_cluster_url
):
    await create_connection(base_url, http_client, scheduler_url=local_cluster_url)
    print("checkpoint 1")

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/"
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url
        )

        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, ca_uuid,
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
async def test_analysis_removal(
    default_raw, base_url, http_client, server_port, shared_state, local_cluster_url
):
    await create_connection(base_url, http_client, scheduler_url=local_cluster_url)

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/"
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url
        )

        # compound analysis is first created without any analyses:
        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, details=None,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, ca_uuid,
        )

        # compound analysis is updated with the newly created analysis:
        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, details={
                "mainType": "APPLY_RING_MASK",
                "analyses": [analysis_uuid]
            }, ca_uuid=ca_uuid
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid
        )

        await consume_task_results(ws, job_uuid)

    # we are done with this analysis, clean up:
    # this will also remove the associated jobs
    async with http_client.delete(analysis_url) as resp:
        assert resp.status == 200
        assert_msg(await resp.json(), 'ANALYSIS_REMOVED')

    assert job_uuid not in shared_state.job_state.jobs
    assert analysis_uuid not in shared_state.analysis_state.analyses

    # also remove dataset:
    async with http_client.delete(ds_url) as resp:
        assert resp.status == 200
        assert_msg(await resp.json(), 'DELETE_DATASET')

    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')
        assert initial_msg == {
            "status": "ok",
            "messageType": "INITIAL_STATE",
            "datasets": [],
            "jobs": [],
            "analyses": [],
            "compoundAnalyses": [],
        }


@pytest.mark.asyncio
async def test_create_compound_analysis(
    default_raw, base_url, http_client, server_port, local_cluster_url
):
    await create_connection(base_url, http_client, scheduler_url=local_cluster_url)

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/"
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url
        )

        # compound analysis is first created without any analyses:
        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, details=None,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, ca_uuid
        )

        # compound analysis is updated with the newly created analysis:
        _, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, details={
                "mainType": "APPLY_RING_MASK",
                "analyses": [analysis_uuid]
            }, ca_uuid=ca_uuid
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid
        )

        await consume_task_results(ws, job_uuid)
