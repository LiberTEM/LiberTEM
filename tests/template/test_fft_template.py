import io
import nbformat
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


def test_sum_fft_default():
    conn = {'connection': {'type': 'local'}}

    dataset = {
        "type": "HDF5",
        "params": {
            "path": "./hdf5_sample.h5",
            "ds_path": "/dataset"
            },
    }

    # TODO: check for better params
    analysis = [{
            "analysisType": 'FFTSUM_FRAMES',
            "parameters": {
                'real_rad': 8,
                'real_centerx': 6,
                'real_centery': 6,
                }
    }]

    notebook = notebook_generator(conn, dataset, analysis)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    # TODO: remove the kernel
    ep = ExecutePreprocessor(timeout=600)
    try:
        out = ep.preprocess(nb, {"metadata": {"path": "."}})
    except CellExecutionError:
        out = None
    assert out is not None


def test_fft_analysis():

    conn = {'connection': {'type': 'local'}}

    dataset = {
        "type": "HDF5",
        "params": {
            "path": "./hdf5_sample.h5",
            "ds_path": "/dataset"
            },
    }

    analysis = [{
            "analysisType": 'APPLY_FFT_MASK',
            "parameters": {
                    'rad_in': 4,
                    'rad_out': 8,
                    'real_rad': 4,
                    'real_centerx': 8,
                    'real_centery': 8
                    }
    }]

    notebook = notebook_generator(conn, dataset, analysis)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    try:
        out = ep.preprocess(nb, {"metadata": {"path": "."}})
    except CellExecutionError:
        out = None
    assert out is not None
