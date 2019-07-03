import numpy as np

from libertem.contrib.dask import make_dask_array

from utils import MemoryDataSet, _mk_random


def test_dask_array():
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(16, 16, 16),
        num_partitions=2,
    )
    (da, workers) = make_dask_array(dataset)
    assert np.allclose(da, data)
    assert np.allclose(da.sum().compute(workers=workers), data.sum())
    assert da.shape == data.shape
