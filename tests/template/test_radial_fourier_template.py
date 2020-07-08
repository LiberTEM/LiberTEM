import io
import nbformat
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


def test_radial_fourier_default():

    conn = {'connection': {'type': 'local'}}

    dataset = {
        "type": "HDF5",
        "params": {
            "path": "./hdf5_sample.h5",
            "ds_path": "/dataset"
            },
    }

    analysis = [{
            "analysisType": "RADIAL_FOURIER",
            "parameters": {
                    'shape': 'radial_fourier',
                    'cx': 0,
                    'cy': 0,
                    'ri': 0,
                    'ro': 10,
                    'n_bins': 2,
                    'max_order': 23,
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
