import io
import os
import numpy as np
import nbformat
from temp_utils import _get_hdf5_params
from libertem.analysis import SumfftAnalysis, ApplyFFTMask, PickFFTFrameAnalysis
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor
import pytest


@pytest.mark.slow
def test_sum_fft_default(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)

    params = {
           'real_rad': 8,
           'real_centerx': 6,
           'real_centery': 6,
    }

    analysis = [{
            "analysisType": 'FFTSUM_FRAMES',
            "parameters": params
    }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'sumfft_result.npy')
    results = np.load(data_path)

    analysis = SumfftAnalysis(
                          dataset=hdf5_ds_2,
                          parameters=params
                        )
    expected = lt_ctx.run(analysis)
    assert np.allclose(
        results,
        expected['intensity'].raw_data,
    )


def test_fft_analysis(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)

    params = {
         'rad_in': 4,
         'rad_out': 8,
         'real_rad': 4,
         'real_centerx': 8,
         'real_centery': 8
    }

    analysis = [{
            "analysisType": 'APPLY_FFT_MASK',
            "parameters": params,
    }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'fft_result.npy')
    results = np.load(data_path)

    analysis = ApplyFFTMask(
                    dataset=hdf5_ds_2,
                    parameters=params
                )
    expected = lt_ctx.run(analysis)
    assert np.allclose(
        results,
        expected['intensity'].raw_data,
    )


def test_pick_fft_analysis(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)

    params = {
        'x': 4,
        'y': 4,
        'real_rad': 4,
        'real_centerx': 8,
        'real_centery': 8,
    }

    analysis = [{
            "analysisType": 'PICK_FFT_FRAME',
            "parameters": params,
    }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'pickfft_result.npy')
    results = np.load(data_path)

    analysis = PickFFTFrameAnalysis(
                    dataset=hdf5_ds_2,
                    parameters=params
                )
    expected = lt_ctx.run(analysis)
    assert np.allclose(
        results,
        expected['intensity'].raw_data,
    )
