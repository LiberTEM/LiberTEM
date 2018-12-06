import os

import numpy as np
import pytest

from libertem import api
from utils import _naive_mask_apply
from libertem.executor.dask import DaskJobExecutor

@pytest.mark.skipif('LT_RUN_FUNCTIONAL' not in os.environ, reason="Takes a long time")
def test_start_local(hdf5_ds_1):
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    with hdf5_ds_1.get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    with api.Context() as ctx:
        analysis = ctx.create_mask_analysis(
            dataset=hdf5_ds_1, factories=[lambda: mask]
        )
        results = ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )

@pytest.mark.skipif('LT_RUN_FUNCTIONAL' not in os.environ, reason="Takes a long time")
def test_subprocess_start_local(hdf5_ds_1):
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    with hdf5_ds_1.get_h5ds() as h5ds:
        data = h5ds[:]
        expected = _naive_mask_apply([mask], data)

    with api.subprocess_create_executor() as ex:
        ctx = api.Context(ex)
        analysis = ctx.create_mask_analysis(
            dataset=hdf5_ds_1, factories=[lambda: mask]
        )
        results = ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected
    )

@pytest.mark.skipif('LT_RUN_FUNCTIONAL' not in os.environ, reason="Takes a long time")
def test_subprocess_errorhandling():
    # Include an argument that makes dask LocalCluster trip
    cluster_kwargs = {
        "threads_per_worker": 1,
        "n_workers": 2,
        "breakme": "die die die"
    }
    try:
        with DaskJobExecutor.subprocess_make_local(cluster_kwargs=cluster_kwargs) as ex:
            # This should not be reached
            ctx = api.Context(ex)
    except Exception as e:
        text = e.args[0]

    # We make sure we get the expected error message
    assert "Starting subprocess failed." in text
