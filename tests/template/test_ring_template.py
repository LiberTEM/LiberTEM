import io
import os
import shutil
import nbformat
import pytest
import numpy as np
from temp_utils import _get_hdf5_params, create_random_hdf5
from libertem.web.notebook_generator.notebook_generator import notebook_generator
from nbconvert.preprocessors import ExecutePreprocessor


pytestmark = [pytest.mark.functional]


def test_ring_default(hdf5_ds_2, tmpdir_factory, lt_ctx):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'local'}}
    path = hdf5_ds_2.path
    dataset = _get_hdf5_params(path)

    analysis = [{
            "analysisType": "APPLY_RING_MASK",
            "parameters": {
                    'shape': 'ring',
                    'cx': 8,
                    'cy': 8,
                    'ri': 5,
                    'ro': 8,
                    }
            }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    out = ep.preprocess(nb, {"metadata": {"path": datadir}})
    data_path = os.path.join(datadir, 'ring_result.npy')
    results = np.load(data_path)

    analysis = lt_ctx.create_ring_analysis(
                            dataset=hdf5_ds_2,
                            cx=8,
                            cy=8,
                            ri=5,
                            ro=8
                        )
    expected = lt_ctx.run(analysis)

    assert np.allclose(
        results,
        expected['intensity'].raw_data,
    )


@pytest.mark.dist
@pytest.mark.asyncio
def test_ring_tcp_cluster(lt_ctx):

    conn = {"connection": {
                    "type": "TCP",
                    "address": "tcp://scheduler:8786"
                    }
            }
    base_path = '/data/temp_data'
    tmp_dir = os.path.join(base_path, 'tmp')
    os.makedirs(tmp_dir)
    ds_path = os.path.join(tmp_dir, 'tmp_random_hdf5.h5')
    ds = create_random_hdf5(ds_path)
    dataset = _get_hdf5_params(ds_path)

    analysis = [{
            "analysisType": "APPLY_RING_MASK",
            "parameters": {
                    'shape': 'ring',
                    'cx': 8,
                    'cy': 8,
                    'ri': 5,
                    'ro': 8,
                    }
            }]

    notebook = notebook_generator(conn, dataset, analysis, save=True)
    notebook = io.StringIO(notebook.getvalue())
    nb = nbformat.read(notebook, as_version=4)
    ep = ExecutePreprocessor(timeout=600)
    out = ep.preprocess(nb, {"metadata": {"path": tmp_dir}})
    data_path = os.path.join(tmp_dir, 'ring_result.npy')
    results = np.load(data_path)

    analysis = lt_ctx.create_ring_analysis(
                            dataset=ds,
                            cx=8,
                            cy=8,
                            ri=5,
                            ro=8
                        )
    expected = lt_ctx.run(analysis)

    assert np.allclose(
        results,
        expected['intensity'].raw_data,
    )
    # cleanup files after test
    shutil.rmtree(tmp_dir, ignore_errors=True)
