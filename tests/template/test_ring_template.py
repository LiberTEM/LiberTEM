import io
import os
import glob

import nbformat
import pytest
import numpy as np
from nbconvert.preprocessors import ExecutePreprocessor

from libertem.web.notebook_generator.notebook_generator import notebook_generator

from utils import get_testdata_path
from temp_utils import _get_hdf5_params, create_random_hdf5


pytestmark = [pytest.mark.slow]

TMP_TESTDATA_PATH = os.path.join(get_testdata_path(), 'temp_data')
HAVE_TMP_TESTDATA = os.path.exists(TMP_TESTDATA_PATH)


def test_ring_default(hdf5_ds_2, tmpdir_factory, lt_ctx, local_cluster_url):
    datadir = tmpdir_factory.mktemp('template_tests')

    conn = {'connection': {'type': 'tcp', 'address': local_cluster_url}}
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
    ep.preprocess(nb, {"metadata": {"path": datadir}})
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


@pytest.fixture(scope='function')
def random_hdf5_1():
    tmp_dir = os.path.join(TMP_TESTDATA_PATH, 'tmp_ring_template')
    os.mkdir(tmp_dir)
    ds_path = os.path.join(tmp_dir, 'tmp_random_hdf5.h5')
    ds = create_random_hdf5(ds_path)
    yield ds
    os.remove(ds_path)
    os.rmdir(tmp_dir)


@pytest.mark.dist
@pytest.mark.asyncio
@pytest.mark.skipif(not HAVE_TMP_TESTDATA, reason="need temporary directory for testdata")  # NOQA
def test_ring_tcp_cluster(lt_ctx, random_hdf5_1, scheduler_addr):

    conn = {"connection": {
                    "type": "TCP",
                    "address": scheduler_addr
                    }
            }
    ds = random_hdf5_1
    ds_path = ds.path
    tmp_dir = os.path.dirname(ds_path)
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
    data_path = None
    try:
        ep.preprocess(nb, {"metadata": {"path": tmp_dir}})
        data_path = os.path.join(tmp_dir, 'ring_result.npy')
        results = np.load(data_path)
    finally:
        if data_path is not None and os.path.exists(data_path):
            os.remove(data_path)

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


@pytest.mark.dist
def test_datadir_inspection():
    print("get_testdata_path():", get_testdata_path())
    print(list(sorted(glob.glob(get_testdata_path() + '/*'))))
    print(list(sorted(glob.glob(TMP_TESTDATA_PATH + '/*'))))
    assert HAVE_TMP_TESTDATA
