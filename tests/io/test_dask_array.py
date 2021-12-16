import numpy as np
import pytest

from libertem.contrib.daskadapter import make_dask_array, task_results_array
from libertem.api import Context
from libertem.io.dataset.memory import MemoryDataSet
from libertem.udf.sum import SumUDF
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.raw import PickUDF

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


@pytest.mark.parametrize(
    'roi', (None, True)
)
def test_dask_results_array(lt_ctx, roi):
    data = _mk_random(size=(13, 14, 15, 16))
    dataset = MemoryDataSet(
        data=data,
        tileshape=(3, 7, 11),
        num_partitions=3,
    )

    if roi:
        roi = np.random.choice([True, False], dataset.shape.nav)

    mask = np.random.random(dataset.shape.sig)

    def factory():
        return mask

    udfs = [PickUDF(), SumUDF(), ApplyMasksUDF(mask_factories=[factory])]

    results = task_results_array(dataset=dataset, udf=udfs, roi=roi)

    ref = lt_ctx.run_udf(dataset=dataset, udf=udfs, roi=roi)

    assert np.allclose(
        # Merge function for PickUDF is sum
        results[0]['intensity'].sum(axis=0).compute(),
        ref[0]['intensity'].raw_data
    )

    assert np.allclose(
        results[1]['intensity'].sum(axis=0).compute(),
        ref[1]['intensity'].raw_data
    )

    assert np.allclose(
        results[2]['intensity'].compute(),
        ref[2]['intensity'].raw_data
    )
