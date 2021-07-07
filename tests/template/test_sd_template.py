import io
import os

import numpy as np
import pytest
import nbformat
from temp_utils import _get_hdf5_params
from libertem.udf.stddev import StdDevUDF
from libertem import masks
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor


@pytest.mark.slow
def test_sd_default(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)
    params = {"roi": {}}
    analysis = [{
            "analysisType": 'SD_FRAMES',
            "parameters": params,
            }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'sd_result.npy')
    results = np.load(data_path)
    udf = StdDevUDF()
    expected = lt_ctx.run_udf(dataset=hdf5_ds_2, udf=udf)
    assert np.allclose(
        results,
        expected['varsum'],
    )


@pytest.mark.slow
def test_sd_roi(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)

    roi_params = {
        "shape": "rect",
        "x": 1,
        "y": 2,
        "width": 6,
        "height": 6
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
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'sd_result.npy')
    results = np.load(data_path)
    nx, ny = hdf5_ds_2.shape.nav
    roi = masks.rectangular(
                X=roi_params["x"],
                Y=roi_params["y"],
                Width=roi_params["width"],
                Height=roi_params["height"],
                imageSizeX=nx,
                imageSizeY=ny)
    udf = StdDevUDF()
    expected = lt_ctx.run_udf(dataset=hdf5_ds_2, udf=udf, roi=roi)
    assert np.allclose(
        results,
        expected['varsum'],
    )
