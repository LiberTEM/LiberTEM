import numpy as np

from utils import MemoryDataSet, _mk_random


def test_dask_array():
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(16, 16, 16),
        num_partitions=2,
    )
    da = dataset.get_dask_array()
    assert np.allclose(da, data)
    assert np.allclose(da.sum().compute(), data.sum())
    assert da.shape == data.shape
