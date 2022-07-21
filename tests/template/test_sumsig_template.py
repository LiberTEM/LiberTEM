import io
import os
import numpy as np
import nbformat
from temp_utils import _get_hdf5_params
from libertem.analysis import SumSigAnalysis
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor
import pytest


@pytest.mark.slow
def test_sum_sig_default(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)

    analysis = [{
            "analysisType": 'SUM_SIG',
            "parameters": {}
    }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'sumsig_result.npy')
    results = np.load(data_path)

    analysis = SumSigAnalysis(
                          dataset=hdf5_ds_2,
                          parameters={}
                        )
    expected = lt_ctx.run(analysis)
    assert np.allclose(
        results,
        expected['intensity'].raw_data,
    )
