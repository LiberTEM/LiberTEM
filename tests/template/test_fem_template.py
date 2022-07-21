import io
import os
import numpy as np
import nbformat
from temp_utils import _get_hdf5_params
from libertem.udf.FEM import FEMUDF
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor
import pytest


@pytest.mark.slow
def test_fem_analysis(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)

    params = {
        'shape': 'ring',
        'cx': 1,
        'cy': 1,
        'ri': 0,
        'ro': 1,
    }

    analysis = [{
            "analysisType": 'FEM',
            "parameters": params
    }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'fem_result.npy')
    results = np.load(data_path)

    analysis = FEMUDF(center=(1, 1), rad_in=0, rad_out=1)
    expected = lt_ctx.run_udf(dataset=hdf5_ds_2, udf=analysis)
    assert np.allclose(
        results,
        expected['intensity'],
    )
