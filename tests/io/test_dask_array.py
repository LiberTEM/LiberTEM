import pytest
import numpy as np

from libertem.contrib.daskadapter import make_dask_array, _flat_slices_for_chunking
from libertem.api import Context
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


@pytest.mark.parametrize(
    "repeat", range(30)
)
def test_flat_slices(repeat):
    """
    take from slices until exhausted and
    accumulate until we exceed or equal prod(shape[-n:])
    assert the accumulator divides exactly by prod(shape[-n:])
    to be sure that the combination of slices is a factor
    repeat for n in range(len(shape)) right-to-left
    assures that all combinations of slices reshape
    across each dimension correctly
    """
    min_blocks = np.random.randint(1, 10)
    ndims = np.random.randint(1, 4)
    shape = np.random.randint(1, 333, size=(ndims,))
    nframes = np.prod(shape)
    target_size = np.random.randint(1, nframes * 2)
    slices = _flat_slices_for_chunking(shape, target_size, min_blocks=min_blocks)
    assert len(slices) >= min_blocks

    for idx in range(1, 1 + len(shape)):
        static_size = np.prod(shape[-idx:])
        acc = 0
        for start, stop in slices:
            acc += (stop - start)
            if acc >= static_size:
                assert acc % static_size == 0
                acc = 0
    assert sum(e - s for s, e in slices) == nframes


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
