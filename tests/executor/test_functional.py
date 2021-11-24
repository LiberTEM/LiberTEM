import pytest
import distributed
import numpy as np

from libertem.executor.delayed import DelayedJobExecutor
from libertem.executor.dask import DaskJobExecutor
from libertem.executor.concurrent import ConcurrentJobExecutor
from libertem.api import Context
from libertem.udf.stddev import StdDevUDF
from libertem.udf.masks import ApplyMasksUDF


@pytest.fixture(
    params=[
        "inline_executor_fast", "dask_executor", "delayed_default",
        "delayed_dist", "threaded_dask_executor", "concurrent"
    ]
)
def executor(request, inline_executor_fast, dask_executor):
    if request.param == 'inline_executor_fast':
        yield inline_executor_fast
    elif request.param == "dask_executor":
        yield dask_executor
    elif request.param == "delayed_default":
        yield DelayedJobExecutor()
    elif request.param == "delayed_dist":
        with distributed.Client(
                n_workers=2,
                threads_per_worker=4,
                processes=True
        ) as _:
            yield DelayedJobExecutor()
    elif request.param == "threaded_dask_executor":
        with distributed.Client(
                n_workers=2,
                threads_per_worker=4,
                processes=False
        ) as c:
            yield DaskJobExecutor(client=c)
    elif request.param == "concurrent":
        yield ConcurrentJobExecutor.make_local()


@pytest.fixture
def ctx(executor):
    return Context(executor=executor)


@pytest.fixture(params=('hdf5', 'raw'))
def ds(request, ctx, hdf5_same_data_4d, raw_same_dataset_4d):
    if request.param == 'hdf5':
        yield ctx.load('HDF5', path=hdf5_same_data_4d.filename)
    elif request.param == 'raw':
        yield ctx.load(
            'RAW',
            path=raw_same_dataset_4d._path,
            nav_shape=raw_same_dataset_4d.shape.nav,
            sig_shape=raw_same_dataset_4d.shape.sig,
            dtype=raw_same_dataset_4d.dtype
        )


@pytest.mark.functional
def test_executors(lt_ctx_fast, ctx, ds):
    def factory():
        m = np.zeros(ds.shape.sig)
        m[-1, -1] = 1.3
        return m

    udfs = [
        StdDevUDF(), ApplyMasksUDF(mask_factories=[factory])
    ]
    ref = lt_ctx_fast.run_udf(dataset=ds, udf=udfs)
    res = ctx.run_udf(dataset=ds, udf=udfs)

    if isinstance(ctx.executor, DelayedJobExecutor):
        res = res.compute()

    assert len(res) == len(ref)
    for i, item in enumerate(ref):
        assert item.keys() == res[i].keys()
        for k in item.keys():
            assert np.allclose(item[k], res[i][k])
