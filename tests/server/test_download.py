import io
import json

import h5py
import pytest
import websockets
import numpy as np

from utils import assert_msg

from aio_utils import (
    create_connection, consume_task_results, create_default_dataset, create_analysis,
    create_job_for_analysis,
)


@pytest.mark.asyncio
async def test_download_1(default_raw, base_url, http_client, server_port):
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
        # this will also remove the associated jobs
        async with http_client.delete(analysis_url) as resp:
            assert resp.status == 200
            assert_msg(await resp.json(), 'ANALYSIS_REMOVED')

        # also get rid of the dataset:
        async with http_client.delete(ds_url) as resp:
            assert resp.status == 200
            resp_json = await resp.json()
            assert_msg(resp_json, 'DELETE_DATASET')
