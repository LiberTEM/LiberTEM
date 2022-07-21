import io
import nbformat
from nbconvert.preprocessors import ExecutePreprocessor
import pytest

from libertem.web.notebook_generator.notebook_generator import notebook_generator
from libertem.io.dataset.base import BufferedBackend


@pytest.mark.slow
def test_custom_io_backend(buffered_raw, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
    path = buffered_raw._path

    dataset_params = {
        "type": "RAW",
        "params": {
            "path": path,
            "dtype": "float32",
            "nav_shape": [16, 16],
            "sig_shape": [128, 128],
            "io_backend": BufferedBackend(),
        }
    }

    analysis = [{
        "analysisType": 'APPLY_DISK_MASK',
        "parameters": {
            'shape': 'disk',
            'cx': 8,
            'cy': 8,
            'r': 5,
        }
    }]

    notebook = notebook_generator(conn, dataset_params, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    # this throws an exception if the `io_backend` parameter is not generated
    # properly in the notebook:
    ep.preprocess(nb, {"metadata": {"path": datadir}})
