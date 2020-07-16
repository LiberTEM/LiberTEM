import io
import os
import nbformat
import numpy as np
from temp_utils import _get_hdf5_params
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor


def test_ring_default(hdf5_ds_1, tmpdir_factory, lt_ctx):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'local'}}
    path = hdf5_ds_1.path
    dataset = _get_hdf5_params(path)

    analysis = [{
            "analysisType": "APPLY_RING_MASK",
            "parameters": {
                    'shape': 'ring',
                    'cx': 8,
                    'cy': 8,
                    'ri': 5,
                    'ro': 8,
                    }
            }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel='libertem-env')
    out = ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'ring_result.npy')
    results = np.load(data_path)

    analysis = lt_ctx.create_ring_analysis(
                            dataset=hdf5_ds_1,
                            cx=8,
                            cy=8,
                            ri=5,
                            ro=8
                        )
    expected = lt_ctx.run(analysis)

    assert np.allclose(
        results,
        expected['intensity'].raw_data,
    )
