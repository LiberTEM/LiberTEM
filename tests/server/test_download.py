import io
import json

import h5py
import pytest
import websockets
import numpy as np
import nbformat

from utils import assert_msg

from aio_utils import (
    create_connection, consume_task_results, create_default_dataset, create_analysis,
    create_job_for_analysis, create_update_compound_analysis,
)
from libertem.io.writers.results import formats  # NOQA: F401
from libertem.io.writers.results.base import ResultFormatRegistry


pytestmark = [pytest.mark.web_api]


@pytest.mark.asyncio
async def test_download_hdf5(
    default_raw, base_url, http_client, server_port, local_cluster_url, default_token,
):
    await create_connection(base_url, http_client, local_cluster_url, token=default_token)

    print("checkpoint 1")

    # connect to ws endpoint:
    ws_url = f"ws://127.0.0.1:{server_port}/api/events/?token={default_token}"
    async with websockets.connect(ws_url) as ws:
        await ws.ensure_open()
        print(ws.state)
        print("checkpoint 2")
        initial_msg = json.loads(await ws.recv())
        assert_msg(initial_msg, 'INITIAL_STATE')

        ds_uuid, ds_url = await create_default_dataset(
            default_raw, ws, http_client, base_url, token=default_token,
        )

        print("checkpoint 3")

        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, token=default_token,
        )

        print("checkpoint 4")

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, ca_uuid, token=default_token,
        )

        print("checkpoint 5")

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid, token=default_token,
        )

        await consume_task_results(ws, job_uuid)

        download_url = "{}/api/compoundAnalyses/{}/analyses/{}/download/HDF5/?token={}".format(
            base_url, ca_uuid, analysis_uuid, default_token,
        )
        async with http_client.get(download_url) as resp:
            assert resp.status == 200
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
        # this will also remove the associated jobs
        async with http_client.delete(analysis_url) as resp:
            assert resp.status == 200
            assert_msg(await resp.json(), 'ANALYSIS_REMOVED')

        # also get rid of the dataset:
        async with http_client.delete(ds_url) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DELETE_DATASET')


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'filetype',
    list(ResultFormatRegistry.get_available_formats().keys())
)
async def test_download_other_formats(
    default_raw, base_url, http_client, server_port, filetype, local_cluster_url, default_token,
):
    """
    This test just triggers download with different formats,
    it does not yet check the result files!
    """
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
            ws, http_client, base_url, ds_uuid, token=default_token,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, ca_uuid, token=default_token,
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid, token=default_token,
        )

        await consume_task_results(ws, job_uuid)

        download_url = "{}/api/compoundAnalyses/{}/analyses/{}/download/{}/?token={}".format(
            base_url, ca_uuid, analysis_uuid, filetype, default_token,
        )
        async with http_client.get(download_url) as resp:
            assert resp.status == 200
            raw_data = await resp.read()
            bio = io.BytesIO(raw_data)
            bio.seek(0)
            assert len(bio.read()) > 0

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

# TODO: add more checks that actually load the resulting files
# TODO: test multi-channel results (most important: HDF5 and TIFF)


@pytest.mark.asyncio
async def test_download_com(
    default_raw, base_url, http_client, server_port, local_cluster_url, default_token,
):
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
                "mainType": "CENTER_OF_MASS",
                "analyses": [],
            }, token=default_token,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, ca_uuid, details={
                "analysisType": "CENTER_OF_MASS",
                "parameters": {
                    "cx": 8,
                    "cy": 8,
                    "r": 32,
                }
            }, token=default_token,
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid, token=default_token,
        )

        await consume_task_results(ws, job_uuid)

        download_url = "{}/api/compoundAnalyses/{}/analyses/{}/download/TIFF/?token={}".format(
            base_url, ca_uuid, analysis_uuid, default_token,
        )
        async with http_client.get(download_url) as resp:
            assert resp.status == 200

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


@pytest.mark.asyncio
async def test_download_notebook(
    default_raw, base_url, http_client, server_port, local_cluster_url, default_token,
):
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
                "mainType": "CENTER_OF_MASS",
                "analyses": [],
            }, token=default_token,
        )

        analysis_uuid, analysis_url = await create_analysis(
            ws, http_client, base_url, ds_uuid, ca_uuid, details={
                "analysisType": "CENTER_OF_MASS",
                "parameters": {
                    "cx": 8,
                    "cy": 8,
                    "r": 32,
                }
            }, token=default_token,
        )

        ca_uuid, ca_url = await create_update_compound_analysis(
            ws, http_client, base_url, ds_uuid, details={
                "mainType": "CENTER_OF_MASS",
                "analyses": [analysis_uuid],
            }, token=default_token,
        )

        job_uuid, job_url = await create_job_for_analysis(
            ws, http_client, base_url, analysis_uuid, token=default_token,
        )

        await consume_task_results(ws, job_uuid)

        download_url = "{}/api/compoundAnalyses/{}/download/notebook/?token={}".format(
            base_url, ca_uuid, default_token,
        )
        async with http_client.get(download_url) as resp:
            assert resp.status == 200
            raw_data = await resp.read()
            notebook = io.BytesIO(raw_data)
            notebook = nbformat.reads(notebook.getvalue(), as_version=4)
            assert nbformat.validate(notebook) is None

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
