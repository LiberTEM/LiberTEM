import os
from glob import glob
import concurrent.futures
import multiprocessing.pool

import pytest
import distributed
import dask
import numpy as np

from libertem.executor.delayed import DelayedJobExecutor
from libertem.executor.dask import DaskJobExecutor
from libertem.executor.concurrent import ConcurrentJobExecutor
from libertem.executor.inline import InlineJobExecutor
from libertem.api import Context
from libertem.udf.stddev import StdDevUDF
from libertem.udf.masks import ApplyMasksUDF

from utils import get_testdata_path


@pytest.fixture(
    params=[
        "inline", "dask_executor", "dask_make_default", "dask_integration",
        "concurrent",
    ]
)
def ctx(request, dask_executor):
    if request.param == 'inline':
        yield Context.make_with('inline')
    elif request.param == "dask_executor":
        yield Context(executor=dask_executor)
    elif request.param == "delayed_default":
        yield Context(executor=DelayedJobExecutor())
    elif request.param == "delayed_dist":
        with distributed.Client(
                n_workers=2,
                threads_per_worker=4,
                processes=True
        ) as _:
            yield Context(executor=DelayedJobExecutor())
    elif request.param == "dask_make_default":
        try:
            ctx = Context.make_with('dask-make-default')
            yield ctx
        finally:
            # cleanup: Close cluster and client
            # This is also tested below, here just to make
            # sure things behave as expected.
            assert isinstance(ctx.executor, DaskJobExecutor)
            ctx.executor.is_local = True
            ctx.close()
    elif request.param == "dask_integration":
        with distributed.Client(
                n_workers=2,
                threads_per_worker=4,
                processes=False
        ) as _:
            yield Context.make_with("dask-integration")
    elif request.param == "concurrent":
        yield Context.make_with("threads")


@pytest.fixture(scope='session')
def load_kwargs(hdf5, default_raw):
    kwargs = [
        {
            'filetype': 'HDF5',
            'path': hdf5.filename
        },
        {
            'filetype': 'RAW',
            'path': default_raw._path,
            'nav_shape': default_raw.shape.nav,
            'sig_shape': default_raw.shape.sig,
            'dtype': default_raw.dtype
        },
        {
            'filetype': 'memory',
            'data': np.ones((3, 4, 5, 6))
        }
    ]
    testdata_path = get_testdata_path()
    blo_path = os.path.join(testdata_path, 'default.blo')
    if os.path.isfile(blo_path):
        kwargs.append({
            'filetype': 'BLO',
            'path': blo_path,
        })
    dm_files = list(sorted(glob(os.path.join(testdata_path, 'dm', '*.dm4'))))
    if dm_files:
        kwargs.append({
            'filetype': 'dm',
            'files': dm_files
        })
    empad_path = os.path.join(testdata_path, 'EMPAD', 'acquisition_12_pretty.xml')
    if os.path.isfile(empad_path):
        kwargs.append({
            'filetype': 'EMPAD',
            'path': empad_path
        })
    frms6_path = os.path.join(testdata_path, 'frms6', 'C16_15_24_151203_019.hdr')
    if os.path.isfile(frms6_path):
        kwargs.append({
            'filetype': 'frms6',
            'path': frms6_path
        })
    k2is_path = os.path.join(testdata_path, 'Capture52', 'Capture52_.gtg')
    if os.path.isfile(k2is_path):
        kwargs.append({
            'filetype': 'k2is',
            'path': k2is_path
        })
    mib_path = os.path.join(testdata_path, 'default.mib')
    if os.path.isfile(mib_path):
        kwargs.append({
            'filetype': 'mib',
            'path': mib_path,
            'nav_shape': (32, 32)
        })
    mrc_path = os.path.join(testdata_path, 'mrc', '20200821_92978_movie.mrc')
    if os.path.isfile(mrc_path):
        kwargs.append({
            'filetype': 'mrc',
            'path': mrc_path
        })
    seq_path = os.path.join(testdata_path, 'default.seq')
    if os.path.isfile(seq_path):
        kwargs.append({
            'filetype': 'seq',
            'path': seq_path,
            'nav_shape': (8, 8)
        })
    ser_path = os.path.join(testdata_path, 'default.ser')
    if os.path.isfile(ser_path):
        kwargs.append({
            'filetype': 'ser',
            'path': ser_path
        })

    return kwargs


def _make_udfs(ds):
    def factory():
        m = np.zeros(ds.shape.sig)
        m[-1, -1] = 1.3
        return m

    udfs = [
        StdDevUDF(), ApplyMasksUDF(mask_factories=[factory])
    ]
    return udfs


def _calculate(ctx, load_kwargs):
    result = {}
    print(f"calculating with {ctx.executor}")
    for kwargs in load_kwargs:
        ds = ctx.load(**kwargs)
        udfs = _make_udfs(ds)
        roi = None
        roi = np.zeros(
            np.prod(ds.shape.nav, dtype=np.int64),
            dtype=bool
        )
        roi[0] = True
        roi[-1] = True
        roi[len(roi)//2] = True
        roi = roi.reshape(ds.shape.nav)
        print(f"calculating {kwargs['filetype']}")
        result[kwargs['filetype']] = ctx.run_udf(
            dataset=ds, udf=udfs, roi=roi
        )
    return result


@pytest.fixture(scope='session')
def reference(load_kwargs):
    ctx = Context(executor=InlineJobExecutor())
    return _calculate(ctx, load_kwargs)


@pytest.mark.slow
def test_executors(ctx, load_kwargs, reference):
    results = _calculate(ctx, load_kwargs)
    for key, res in results.items():
        print(f"filetype: {key}")
        if isinstance(ctx.executor, DelayedJobExecutor):
            res = res.compute()

        assert len(res) == len(reference[key])
        for i, item in enumerate(reference[key]):
            assert item.keys() == res[i].keys()
            for buf_key in item.keys():
                print(f"buffer {buf_key}")
                left = item[buf_key].raw_data
                right = res[i][buf_key].raw_data
                print(np.max(np.abs(left - right)))
                print(np.min(np.abs(left)))
                print(np.min(np.abs(right)))
                assert np.allclose(left, right)


@pytest.mark.slow
def test_make_default():
    try:
        ctx = Context.make_with("dask-make-default")
        # This queries Dask which scheduler it is using
        ctx2 = Context.make_with("dask-integration")
        # make sure the second uses the Client of the first
        assert ctx2.executor.client is ctx.executor.client
    finally:
        # dask-make-default starts a Client that will persist
        # and not be closed automatically. We have to make sure
        # to close everything ourselves
        if ctx.executor.client.cluster is not None:
            ctx.executor.client.cluster.close(timeout=30)
        ctx.executor.client.close()
        ctx.close()


def test_connect_default(local_cluster_url):
    try:
        executor = DaskJobExecutor.connect(
            local_cluster_url,
            client_kwargs={'set_as_default': True}
        )
        ctx = Context(executor=executor)
        # This queries Dask which scheduler it is using
        ctx2 = Context.make_with("dask-integration")
        # make sure the second uses the Client of the first
        assert ctx2.executor.client is ctx.executor.client
    finally:
        # Only close the Client, keep the cluster running
        # since that is test infrastructure
        executor.client.close()
        ctx.close()


def test_use_distributed():
    # This Client is pretty cheap to start
    # since it only uses threads
    with distributed.Client(
        n_workers=1, threads_per_worker=1, processes=False
    ) as c:
        ctx = Context.make_with("dask-integration")
        assert isinstance(ctx.executor, DaskJobExecutor)
        assert ctx.executor.client is c


def test_no_dangling_client():
    # Within the whole test suite and LiberTEM we should not have
    # a dangling dask.distributed Client set as default Dask scheduler.
    # That means we confirm that we get a ConcurrentJobExecutor in the
    # default case.
    ctx = Context.make_with("dask-integration")
    assert isinstance(ctx.executor, ConcurrentJobExecutor)


def test_use_threads():
    with dask.config.set(scheduler="threads"):
        ctx = Context.make_with("dask-integration")
        assert isinstance(ctx.executor, ConcurrentJobExecutor)
        assert isinstance(
            ctx.executor.client,
            (
                concurrent.futures.ThreadPoolExecutor,
                multiprocessing.pool.ThreadPool,
            )
        )


def test_use_synchronous():
    with dask.config.set(scheduler="synchronous"):
        ctx = Context.make_with("dask-integration")
        assert isinstance(ctx.executor, InlineJobExecutor)
