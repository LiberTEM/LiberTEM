import numpy as np
import distributed as dd

from libertem.contrib.daskadapter import make_dask_array
from libertem.api import Context
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


def test_dask_array():
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(16, 16, 16),
        num_partitions=2,
    )
    (da, workers) = make_dask_array(dataset)
    assert np.allclose(
        da.compute(workers=workers, scheduler='single-threaded'),
        data
    )
    assert np.allclose(
        da.sum().compute(workers=workers, scheduler='single-threaded'),
        data.sum()
    )
    assert da.shape == data.shape


def test_dask_array_2(dask_executor):
    # NOTE: keep in sync with the example in docs/source/api.rst!
    # Construct a Dask array from the dataset
    # The second return value contains information
    # on workers that hold parts of a dataset in local
    # storage to ensure optimal data locality
    ctx = Context(executor=dask_executor)
    dataset = ctx.load("memory", datashape=(16, 16, 16), sig_dims=2)
    dask_array, workers = make_dask_array(dataset)

    # Use the Dask.distributed client of LiberTEM, since it may not be
    # the default client:
    result = ctx.executor.client.compute(
        dask_array.sum(axis=(-1, -2))
    ).result()

