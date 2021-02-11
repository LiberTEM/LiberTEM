import numpy as np

from libertem.contrib.daskadapter import make_dask_array

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


def test_dask_array_with_roi():
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(16, 16, 16),
        num_partitions=2,
    )
    roi = np.zeros(dataset.shape.nav, dtype=np.bool)
    roi[0, 0] = True
    (da, workers) = make_dask_array(dataset, roi=roi)
    assert np.allclose(
        da.compute(workers=workers, scheduler='single-threaded'),
        data[0, 0]
    )
    assert da.shape == data[0, 0].shape
