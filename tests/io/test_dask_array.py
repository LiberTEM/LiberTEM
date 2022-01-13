import numpy as np

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
    ctx.executor.client.compute(
        dask_array.sum(axis=(-1, -2))
    ).result()


def test_dask_array_with_roi_1():
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(16, 16, 16),
        num_partitions=2,
    )
    roi = np.zeros(dataset.shape.nav, dtype=bool)
    roi[0, 0] = True
    (da, workers) = make_dask_array(dataset, roi=roi)
    assert np.allclose(
        da.compute(workers=workers, scheduler='single-threaded'),
        data[0, 0]
    )
    assert da.shape == (1, 16, 16)


def test_dask_array_with_roi_2():
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(16, 16, 16),
        num_partitions=2,
    )

    sparse_roi = np.random.choice([True, False], size=dataset.shape.nav, p=[0.1, 0.9])
    (da, workers) = make_dask_array(dataset, roi=sparse_roi)
    assert np.allclose(
        da.compute(workers=workers, scheduler='single-threaded'),
        data[sparse_roi]
    )
    assert da.shape == (np.count_nonzero(sparse_roi), 16, 16)
