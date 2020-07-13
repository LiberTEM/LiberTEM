import os
import numpy as np
import io
import nbformat
from temp_utils import _get_hdf5_params
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from libertem.analysis.getroi import get_roi
from nbconvert.preprocessors import ExecutePreprocessor


def test_sum_default(hdf5_ds_1, tmpdir_factory):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'local'}}
    path = hdf5_ds_1.path
    dataset = _get_hdf5_params(path)
    analysis = [{
            "analysisType": "SUM_FRAMES",
            "parameters": {
                    "roi": {}
                    }
            }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel='libertem-env')
    out = ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'sum_result.npy')
    result = np.load(data_path)
    with hdf5_ds_1.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
    expected = data.sum(axis=(0, 1))
    assert np.allclose(expected, result)


def test_sum_roi(hdf5_ds_1, tmpdir_factory, lt_ctx):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'local'}}
    path = hdf5_ds_1.path
    dataset = _get_hdf5_params(path)
    roi_params = {
        "shape": "disk",
        "cx": 8,
        "cy": 8,
        "r": 6
    }
    analysis = [{
                "analysisType": "SUM_FRAMES",
                "parameters": {
                            "roi": roi_params
                            }
                }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel='libertem-env')
    out = ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'sum_result.npy')
    results = np.load(data_path)

    analysis = lt_ctx.create_sum_analysis(
                            dataset=hdf5_ds_1,
                        )
    roi = get_roi(roi_params, hdf5_ds_1.shape.nav)
    udf = analysis.get_udf()
    expected = lt_ctx.run_udf(hdf5_ds_1, udf, roi)

    assert np.allclose(
        results,
        expected['intensity'].data,
    )
