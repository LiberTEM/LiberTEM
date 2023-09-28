import json
import pytest
import websockets
import nbformat as nbf

from utils import assert_msg
from nbconvert.preprocessors import ExecutePreprocessor

from aio_utils import (
    create_connection, consume_task_results, create_default_dataset, create_analysis,
    create_job_for_analysis, create_update_compound_analysis,
)
from libertem.io.writers.results import formats  # NOQA: F401


pytestmark = [pytest.mark.web_api]


@pytest.mark.asyncio
async def test_copy_notebook(
    default_raw, base_url, tmpdir_factory, http_client, server_port, local_cluster_url,
    default_token,
):
    datadir = tmpdir_factory.mktemp('test_copy')

    await create_connection(base_url, http_client, local_cluster_url, default_token)

    print("checkpoint 1")

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/?token={default_token}"
    async with websockets.connect(ws_url) as ws:
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url, token=default_token,
        )

        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, details={
                "mainType": "APPLY_RING_MASK",
                "analyses": [],
            }, token=default_token,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, ca_uuid, details={
                "analysisType": "APPLY_RING_MASK",
                "parameters": {
                    "shape": "ring",
                    "cx": 8,
                    "cy": 8,
                    "ri": 5,
                    "ro": 8,
                }
            }, token=default_token,
        )

        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, details={
                "mainType": "APPLY_RING_MASK",
                "analyses": [analysis_uuid],
            }, token=default_token,
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid, token=default_token,
        )

        await consume_task_results(ws, job_uuid)
        download_url = "{}/api/compoundAnalyses/{}/copy/notebook/?token={}".format(
            base_url, ca_uuid, default_token,
        )
        async with http_client.get(download_url) as resp:
            assert resp.status == 200
            code = await resp.json()
            assert 'dependency' in code
            assert 'initial_setup' in code
            assert 'ctx' in code
            assert 'dataset' in code
            assert 'analysis' in code['analysis'][0]
            assert 'plot' in code['analysis'][0]

        # we are done with this analysis, clean up:
        # this will also remove the associated jobs
        async with http_client.delete(analysis_url) as resp:
            assert resp.status == 200
            assert_msg(await resp.json(), 'ANALYSIS_REMOVED')

        # also get rid of the dataset:
        async with http_client.delete(ds_url) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DELETE_DATASET')

        code = "\n".join([
                    code['dependency'],
                    code['initial_setup'],
                    code['ctx'],
                    code['dataset'],
                    code['analysis'][0]['analysis'],
                    code['analysis'][0]['plot'][0]
                ])
        nb = nbf.v4.new_notebook()
        cell = nbf.v4.new_code_cell(code)
        nb['cells'].append(cell)
        ep = ExecutePreprocessor(timeout=600)
        ep.preprocess(nb, {"metadata": {"path": datadir}})
