import os
import numpy as np
import io
import nbformat
from temp_utils import _get_hdf5_params
from libertem.udf.sum import SumUDF
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from libertem import masks
from nbconvert.preprocessors import ExecutePreprocessor
import pytest


@pytest.mark.slow
def test_sum_default(hdf5_ds_2, tmpdir_factory, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
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
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'sum_result.npy')
    result = np.load(data_path)
    with hdf5_ds_2.get_reader().get_h5ds() as h5ds:
        data = h5ds[:]
    expected = data.sum(axis=(0, 1))
    assert np.allclose(expected, result)


@pytest.mark.slow
def test_sum_roi(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
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
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'sum_result.npy')
    results = np.load(data_path)
    nx, ny = hdf5_ds_2.shape.nav
    roi = masks.circular(
                    centerX=roi_params['cx'],
                    centerY=roi_params['cy'],
                    imageSizeX=nx,
                    imageSizeY=ny,
                    radius=roi_params['r'],
            )
    udf = SumUDF()
    expected = lt_ctx.run_udf(hdf5_ds_2, udf, roi)

    assert np.allclose(
        results,
        expected['intensity'].raw_data,
    )
