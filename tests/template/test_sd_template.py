import io
import os
import numpy as np
import nbformat
from temp_utils import _get_hdf5_params
from libertem.udf.stddev import StdDevUDF
from libertem.analysis.getroi import get_roi
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor


def test_sd_default(hdf5_ds_1, tmpdir_factory, lt_ctx):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'local'}}
    path = hdf5_ds_1.path
    dataset = _get_hdf5_params(path)
    params = {"roi": {}}
    analysis = [{
            "analysisType": 'SD_FRAMES',
            "parameters": params,
            }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel='libertem-env')
    out = ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'sd_result.npy')
    results = np.load(data_path)

    # analysis = SDAnalysis(
    #             dataset=hdf5_ds_1,
    #             parameters=params
    #             )
    roi = get_roi(params, hdf5_ds_1.shape.nav)
    udf = StdDevUDF()
    expected = lt_ctx.run_udf(dataset=hdf5_ds_1, udf=udf, roi=roi)
    assert np.allclose(
        results,
        expected['varsum'].raw_data,
    )


def test_sd_roi(hdf5_ds_1, tmpdir_factory, lt_ctx):
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
                "analysisType": 'SD_FRAMES',
                "parameters": {
                            "roi": roi_params
                            }
                }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600, kernel='libertem-env')
    out = ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'sd_result.npy')
    results = np.load(data_path)
    roi = get_roi(roi_params, hdf5_ds_1.shape.nav)
    udf = StdDevUDF()
    expected = lt_ctx.run_udf(dataset=hdf5_ds_1, udf=udf, roi=roi)
    # analysis = SDAnalysis(
    #                 dataset=hdf5_ds_1,
    #                 parameters={'roi': roi_params}
    #             )
    # expected = lt_ctx.run(analysis)
    assert np.allclose(
        results,
        expected['varsum'].raw_data,
    )
