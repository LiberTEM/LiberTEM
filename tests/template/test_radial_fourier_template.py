import io
import os

import pytest
import numpy as np
import nbformat
from temp_utils import _get_hdf5_params
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.mark.slow
def test_radial_fourier_default(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)

    analysis = [{
            "analysisType": "RADIAL_FOURIER",
            "parameters": {
                    'shape': 'radial_fourier',
                    'cx': 0,
                    'cy': 0,
                    'ri': 0,
                    'ro': 2,
                    'n_bins': 2,
                    'max_order': 7,
                    }
    }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    channels = [
        "absolute_0_0",
        "absolute_0_1"
    ]
    results = {}
    for channel in channels:
        data_path = os.path.join(datadir, f"radial_result_{channel}.npy")
        results[channel] = np.load(data_path)

    analysis = lt_ctx.create_radial_fourier_analysis(
                                        dataset=hdf5_ds_2,
                                        cx=0,
                                        cy=0,
                                        ri=0,
                                        ro=2,
                                        n_bins=2,
                                        max_order=7
                                )
    expected = lt_ctx.run(analysis)

    assert np.allclose(results["absolute_0_0"], expected["absolute_0_0"].raw_data)
    assert np.allclose(results["absolute_0_1"], expected["absolute_0_1"].raw_data)
