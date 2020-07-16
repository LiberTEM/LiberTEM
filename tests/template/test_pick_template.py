import os
import numpy as np
import io
import nbformat
from temp_utils import _get_hdf5_params
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor


def test_pick_analysis(hdf5_ds_1, tmpdir_factory, lt_ctx):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'local'}}
    path = hdf5_ds_1.path
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
    ep = ExecutePreprocessor(timeout=600, kernel='libertem-env')
    out = ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'pick_result.npy')
    results = np.load(data_path)

    analysis = lt_ctx.create_pick_analysis(
                                    dataset=hdf5_ds_1,
                                    x=4,
                                    y=4,
                                )
    expected = lt_ctx.run(analysis)
    assert np.allclose(
        results,
        expected['intensity'].raw_data,
    )
