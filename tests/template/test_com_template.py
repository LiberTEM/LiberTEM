import io
import nbformat
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor, CellExecutionError


def test_com_default():
    conn = {'connection': {'type': 'local'}}

    dataset = {
        "type": "HDF5",
        "params": {
            "path": "./hdf5_sample.h5",
            "ds_path": "/dataset"
            },
    }

    analysis = [{
            "analysisType": 'CENTER_OF_MASS',
            "parameters": {
                        'shape': 'com',
                        'cx': 0,
                        'cy': 0,
                        'r': 8,
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
