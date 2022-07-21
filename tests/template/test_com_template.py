import io
import os

import numpy as np
import nbformat
from temp_utils import _get_hdf5_params
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor
import pytest


@pytest.mark.slow
def test_com_default(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)

    analysis = [{
        "analysisType": 'CENTER_OF_MASS',
        "parameters": {
            'shape': 'com',
            'cx': 0,
            'cy': 0,
            'r': 8,
            'ri': 4,
        }
    }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=30)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    channels = [
        "field",
        "magnitude",
        "divergence",
        "curl",
        "x",
        "y"
    ]
    results = {}
    for channel in channels:
        data_path = os.path.join(datadir, f"com_result_{channel}.npy")
        results[channel] = np.load(data_path)

    com_analysis = lt_ctx.create_com_analysis(
        dataset=hdf5_ds_2,
        cx=0,
        cy=0,
        mask_radius=8,
        mask_radius_inner=4,
    )
    expected = lt_ctx.run(com_analysis)

    assert np.allclose(results["field"], expected["field"].raw_data)
    assert np.allclose(results["magnitude"], expected["magnitude"].raw_data)
    assert np.allclose(results["divergence"], expected["divergence"].raw_data)
    assert np.allclose(results["curl"], expected["curl"].raw_data)
    assert np.allclose(results["x"], expected["x"].raw_data)
    assert np.allclose(results["y"], expected["y"].raw_data)


@pytest.mark.slow
def test_com_rotflip(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)

    analysis = [{
            "analysisType": 'CENTER_OF_MASS',
            "parameters": {
                        'shape': 'com',
                        'cx': 0,
                        'cy': 0,
                        'r': 8,
                        'flip_y': True,
                        'scan_rotation': -42.23
                        }
    }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    channels = [
            "field",
            "magnitude",
            "divergence",
            "curl",
            "x",
            "y"
    ]
    results = {}
    for channel in channels:
        data_path = os.path.join(datadir, f"com_result_{channel}.npy")
        results[channel] = np.load(data_path)

    com_analysis = lt_ctx.create_com_analysis(
                                dataset=hdf5_ds_2,
                                cx=0,
                                cy=0,
                                mask_radius=8,
                                flip_y=True,
                                scan_rotation=-42.23,
                            )
    expected = lt_ctx.run(com_analysis)

    assert np.allclose(results["field"], expected["field"].raw_data)
    assert np.allclose(results["magnitude"], expected["magnitude"].raw_data)
    assert np.allclose(results["divergence"], expected["divergence"].raw_data)
    assert np.allclose(results["curl"], expected["curl"].raw_data)
    assert np.allclose(results["x"], expected["x"].raw_data)
    assert np.allclose(results["y"], expected["y"].raw_data)
