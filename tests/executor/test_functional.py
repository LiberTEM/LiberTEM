import os
import sys
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
    scope='session',
    params=[
        "inline", "dask_executor", "dask_make_default", "dask_integration",
        "concurrent", "delayed", "pipelined",
    ]
)
def ctx(request, local_cluster_url):
    print(f'Make ctx with {request.param}')
    if request.param == 'inline':
        yield Context.make_with('inline')
    elif request.param == "dask_executor":
        dask_executor = DaskJobExecutor.connect(local_cluster_url)
        yield Context(executor=dask_executor)
        dask_executor.close()
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
    elif request.param == "delayed":
        yield Context(executor=DelayedJobExecutor())
    elif request.param == "pipelined":
        if sys.version_info < (3, 8):
            pytest.skip("PipelinedExecutor only supported from Python 3.8 onwards")
        else:
            from libertem.executor.pipelined import PipelinedExecutor
            try:
                ctx = Context(executor=PipelinedExecutor(n_workers=2))
                yield ctx
            finally:
                ctx.close()


@pytest.fixture(
    scope='session',
    params=[
        "HDF5", "RAW", "memory", "NPY", "BLO",
        "DM", "EMPAD", "FRMS6", "K2IS",
        "MIB", "MRC", "SEQ", "SER", "TVIPS"
    ]
)
def load_kwargs(request, hdf5, default_raw, default_npy, default_npy_filepath):
    param = request.param
    testdata_path = get_testdata_path()
    if param == 'HDF5':
        return {
            'filetype': 'HDF5',
            'path': hdf5.filename,
        }
    elif param == 'RAW':
        return {
            'filetype': 'RAW',
            'path': default_raw._path,
            'nav_shape': default_raw.shape.nav,
            'sig_shape': default_raw.shape.sig,
            'dtype': default_raw.dtype,
        }
    elif param == 'memory':
        return {
            'filetype': 'memory',
            'data': np.ones((3, 4, 5, 6)),
        }
    elif param == 'NPY':
        return {
            'filetype': 'NPY',
            'path': default_npy_filepath,
        }
    elif param == 'BLO':
        blo_path = os.path.join(testdata_path, 'default.blo')
        if os.path.isfile(blo_path):
            return {
                'filetype': 'BLO',
                'path': blo_path,
            }
    elif param == 'DM':
        dm_files = list(sorted(glob(os.path.join(testdata_path, 'dm', '*.dm4'))))
        if dm_files:
            return {
                'filetype': 'dm',
                'files': dm_files
            }
    elif param == 'EMPAD':
        empad_path = os.path.join(testdata_path, 'EMPAD', 'acquisition_12_pretty.xml')
        if os.path.isfile(empad_path):
            return {
                'filetype': 'EMPAD',
                'path': empad_path
            }
    elif param == 'FRMS6':
        frms6_path = os.path.join(testdata_path, 'frms6', 'C16_15_24_151203_019.hdr')
        if os.path.isfile(frms6_path):
            return {
                'filetype': 'frms6',
                'path': frms6_path
            }
    elif param == 'K2IS':
        k2is_path = os.path.join(testdata_path, 'Capture52', 'Capture52_.gtg')
        if os.path.isfile(k2is_path):
            return {
                'filetype': 'k2is',
                'path': k2is_path
            }
    elif param == 'MIB':
        mib_path = os.path.join(testdata_path, 'default.mib')
        if os.path.isfile(mib_path):
            return {
                'filetype': 'mib',
                'path': mib_path,
                'nav_shape': (32, 32)
            }
    elif param == 'MRC':
        mrc_path = os.path.join(testdata_path, 'mrc', '20200821_92978_movie.mrc')
        if os.path.isfile(mrc_path):
            return {
                'filetype': 'mrc',
                'path': mrc_path
            }
    elif param == 'SEQ':
        seq_path = os.path.join(testdata_path, 'default.seq')
        if os.path.isfile(seq_path):
            return {
                'filetype': 'seq',
                'path': seq_path,
                'nav_shape': (8, 8)
            }
    elif param == 'SER':
        ser_path = os.path.join(testdata_path, 'default.ser')
        if os.path.isfile(ser_path):
            return {
                'filetype': 'ser',
                'path': ser_path
            }
    elif param == 'TVIPS':
        tvips_path = os.path.join(testdata_path, 'TVIPS', 'rec_20200623_080237_000.tvips')
        if os.path.isfile(tvips_path):
            return {
                'filetype': 'TVIPS',
                'path': tvips_path
            }
    else:
        raise ValueError(f'Unknown file type {param}')
    pytest.skip(f"Data file not found for {param}")


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
    print(f"calculating with {ctx.executor}")
    ds = ctx.load(**load_kwargs)
    udfs = _make_udfs(ds)
    roi = np.zeros(
        np.prod(ds.shape.nav, dtype=np.int64),
        dtype=bool
    )
    roi[0] = True
    roi[-1] = True
    roi[len(roi)//2] = True
    roi = roi.reshape(ds.shape.nav)
    print(f"calculating {load_kwargs['filetype']}")
    result = ctx.run_udf(
        dataset=ds, udf=udfs, roi=roi
    )
    return result


@pytest.fixture(scope='session')
def reference(load_kwargs):
    ctx = Context(executor=InlineJobExecutor())
    return _calculate(ctx, load_kwargs)


@pytest.mark.slow
def test_executors(ctx, load_kwargs, reference):
    result = _calculate(ctx, load_kwargs)
    print(f"filetype: {load_kwargs['filetype']}")

    for i, item in enumerate(reference):
        assert item.keys() == result[i].keys()
        for buf_key in item.keys():
            print(f"buffer {buf_key}")
            left = item[buf_key].raw_data
            right = np.array(result[i][buf_key].raw_data)
            print(np.max(np.abs(left - right)))
            print(np.min(np.abs(left)))
            print(np.min(np.abs(right)))
            # To see what type we actually test
            print("is allclose", np.allclose(left, right))
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
