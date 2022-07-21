import os
import numpy as np
import io
import nbformat
from temp_utils import _get_hdf5_params
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor
import pytest


@pytest.mark.slow
def test_pick_analysis(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}

    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)

    analysis = [{
            "analysisType": 'PICK_FRAME',
            "parameters": {
                        'x': 4,
                        'y': 4,
                        }
    }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'pick_result.npy')
    results = np.load(data_path)

    analysis = lt_ctx.create_pick_analysis(
                                    dataset=hdf5_ds_2,
                                    x=4,
                                    y=4,
                                )
    expected = lt_ctx.run(analysis)
    assert np.allclose(
        results,
        expected['intensity'].raw_data,
    )
