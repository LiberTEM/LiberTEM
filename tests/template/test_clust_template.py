import io
import os
import numpy as np
import nbformat
import pytest
from temp_utils import _get_hdf5_params
from libertem.analysis.clust import ClusterAnalysis
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor
from libertem.executor.base import AsyncAdapter


class ResultContainer:
    async def __call__(self, results, finished):
        self.results = results
        self.finished = finished


@pytest.mark.slow
@pytest.mark.asyncio
async def test_clust_default(
    hdf5_ds_2, tmpdir_factory, inline_executor, local_cluster_url,
):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)

    analysis = [{
            "analysisType": 'CLUST',
            "parameters": {
                    'n_peaks': 42,
                    'n_clust': 7,
                    'cy': 3,
                    'cx': 3,
                    'ri': 1,
                    'ro': 5,
                    'delta': 0.05,
                    'min_dist': 1,
                    }
    }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'cluster_result.npy')
    results = np.load(data_path)

    executor = AsyncAdapter(wrapped=inline_executor)
    analysis = ClusterAnalysis(dataset=hdf5_ds_2, parameters={
        'n_peaks': 42,
        'n_clust': 7,
        'cy': 3,
        'cx': 3,
        'ri': 1,
        'ro': 5,
        'delta': 0.05,
        'min_dist': 1,
    })

    uuid = 'bd3b39fb-0b34-4a45-9955-339da6501bbb'

    res_container = ResultContainer()

    async def send_results(results, finished):
        pass

    await analysis.controller(
        cancel_id=uuid, executor=executor,
        job_is_cancelled=lambda: False,
        send_results=res_container,
    )
    expected = res_container.results

    assert np.allclose(
        results,
        expected['intensity'].raw_data
    )
