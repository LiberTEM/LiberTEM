import os
import time
from glob import glob
import concurrent.futures
import multiprocessing.pool
from typing import Any

import pytest
import distributed
import dask
import numpy as np
from libertem.common.shape import Shape
from libertem.common.slice import Slice

from libertem.utils.devices import detect
from libertem.executor.delayed import DelayedJobExecutor
from libertem.executor.dask import DaskJobExecutor
from libertem.executor.concurrent import ConcurrentJobExecutor
from libertem.executor.inline import InlineJobExecutor
from libertem.common.executor import (
    Environment, TaskCommHandler, TaskProtocol, WorkerQueue,
)
from libertem.api import Context
from libertem.udf.stddev import StdDevUDF
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf.sum import SumUDF
from libertem.common.exceptions import ExecutorSpecException
from sparseconverter import (
    BACKENDS, CUPY, CUPY_BACKENDS, CUPY_SCIPY_CSR, NUMPY, SCIPY_COO, SPARSE_COO
)

from utils import get_testdata_path


d = detect()
has_cupy = d['cudas'] and d['has_cupy']
has_gpus = len(d['cudas']) > 0


@pytest.fixture(
    scope='module',
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
            ctx = Context.make_with(
                'dask-make-default',
                cpus=4,
            )
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
        ctx = None
        try:
            ctx = Context.make_with('pipelined', cpus=range(2))
            yield ctx
        finally:
            if ctx is not None:
                ctx.close()


@pytest.fixture(
    scope='function',
    params=[
        # Only testing a selection of backends to keep the number of tests under control
        "HDF5", "RAW", "memory", SCIPY_COO, SPARSE_COO, CUPY, CUPY_SCIPY_CSR, "NPY", "BLO",
        "DM", "EMPAD", "FRMS6", "K2IS",
        "MIB", "MRC", "SEQ", "SER", "TVIPS",
        "RAW_CSR",
    ]
)
def load_kwargs(request, hdf5, default_raw, default_npy, default_npy_filepath, raw_csr_generated):
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
    elif param in BACKENDS:
        if param in CUPY_BACKENDS and not has_cupy:
            pytest.skip("No CuPy, skipping CuPy-based tiles")
        return {
            'filetype': 'memory',
            'data': np.ones((3, 4, 5, 6)),
            'array_backends': (param, NUMPY)
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
        dm_files = list(sorted(glob(os.path.join(testdata_path, 'dm', '2018-7-17*.dm4'))))
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
    elif param == 'RAW_CSR':
        if os.path.isfile(raw_csr_generated._path):
            return {
                'filetype': 'raw_csr',
                'path': raw_csr_generated._path,
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
        dataset=ds, udf=udfs, roi=roi, progress=True,
    )
    return result, udfs


@pytest.fixture(scope='function')
def reference(load_kwargs):
    # This must be function-scoped because load_kwargs is
    # Could refactor the tests to avoid this recalculate if needed
    ctx = Context(executor=InlineJobExecutor())
    return _calculate(ctx, load_kwargs)


@pytest.mark.slow
def test_executors(ctx, load_kwargs, reference):
    result, udfs = _calculate(ctx, load_kwargs)
    print(f"filetype: {load_kwargs['filetype']}")

    for i, item in enumerate(reference[0]):
        print(f"{i}: UDF {udfs[i]}")
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
            print("dtypes", left.dtype, right.dtype)
            assert np.allclose(left, right), \
                f"mismatching result for buffer {buf_key} in UDF {udfs[i]}"


@pytest.mark.slow
@pytest.mark.parametrize(
    'flip', (True, False)
)
def test_tuple_list(flip, lt_ctx, load_kwargs):
    default = lt_ctx.load(**load_kwargs)
    sig_shape = default.shape.sig.to_tuple()
    nav_shape = default.shape.nav.to_tuple()
    if flip:
        sig_shape = list(sig_shape)
    else:
        nav_shape = list(nav_shape)
    load_kwargs['nav_shape'] = nav_shape
    load_kwargs['sig_shape'] = sig_shape
    ds = lt_ctx.load(**load_kwargs)
    lt_ctx.run_udf(dataset=ds, udf=SumUDF())


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


@pytest.mark.slow
def test_make_dask_implicit():
    with Context.make_with(cpus=(4, 7)) as ctx:
        assert ctx.executor.__class__ == DaskJobExecutor
        assert ctx.executor.client is not None
        assert len(ctx.executor.get_available_workers().has_cpu()) == 2


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


# not implemented in most executors, also only used for the clustered / caching
# dataset stuff, which we are not using
@pytest.mark.xfail
@pytest.mark.slow
def test_executor_run_each_partition(ctx: Context):
    ds = ctx.load("memory", data=np.random.randint(0, 1024, size=(16, 16, 128, 128)))
    partitions = ds.get_partitions()

    def _per_partition(p):
        assert p.__class__.__name__ == "MemPartition"
        return 42

    for res in ctx.executor.run_each_partition(partitions, _per_partition):
        assert res == 42


@pytest.mark.slow
def test_executor_map(ctx: Context):
    inp = [1, 2, 3]
    exp = [2, 3, 4]
    assert list(ctx.executor.map(lambda x: x + 1, inp)) == exp


@pytest.mark.slow
def test_executor_run_each_host(ctx: Context):
    res = ctx.executor.run_each_host(lambda x: 42 + x, 1)
    workers = ctx.executor.get_available_workers()
    assert set(res.keys()) == workers.hosts()
    for k, v in res.items():
        assert k in workers.hosts()
        assert v == 43


@pytest.mark.skipif(has_gpus, reason='Test to check error on no-GPU avail')
def test_make_with_no_gpu():
    with pytest.raises(ExecutorSpecException):
        Context.make_with('dask', gpus=2)


@pytest.mark.slow
def test_make_with_scenarios():
    """
    Comment from @uellue on #1443:

    I see several scenarios to use these parameters:

    1. CPU defaults are OK, but I want to hand-craft and optimize my GPU worker setup
    2. GPU defaults are OK/not applicable, but I need to optimize my CPU worker setup.
    3. Co-optimize CPU and GPU setup
    4. Disable all CPU or GPU workers.

    For 1. and 2. it would be good if the user doesn't have to specify the CPU or
    GPU setup to retain it at the defaults. For 4. it is easy to specify ...=0. For
    that reason it would be better, IMO, if leaving a parameter at "None" keeps it
    at the defaults.
    """
    # 1. CPU defaults are OK, but I want to hand-craft and optimize my GPU worker setup
    try:
        with Context.make_with('dask', gpus=2) as ctx:
            assert len(ctx.executor.get_available_workers().has_cpu()) > 0
            assert len(ctx.executor.get_available_workers().has_cuda()) == 2
    except ExecutorSpecException:
        assert not has_gpus, "should only happen if we have no GPUs available"

    # 2. GPU defaults are OK/not applicable, but I need to optimize my CPU worker setup.
    with Context.make_with('dask', cpus=2) as ctx:
        assert len(ctx.executor.get_available_workers().has_cpu()) == 2
        if has_gpus:
            assert len(ctx.executor.get_available_workers().has_cuda()) > 0

    # 3. Co-optimize CPU and GPU setup
    try:
        with Context.make_with('dask', cpus=2, gpus=2) as ctx:
            assert len(ctx.executor.get_available_workers().has_cpu()) > 0
            assert len(ctx.executor.get_available_workers().has_cuda()) == 2
    except ExecutorSpecException:
        assert not has_gpus, "should only happen if we have no GPUs available"

    # 4.1 Disable all CPU* or GPU workers.
    with Context.make_with('dask', cpus=0) as ctx:
        assert len(ctx.executor.get_available_workers().has_cpu()) == 0
        if has_gpus:
            assert len(ctx.executor.get_available_workers().has_cuda()) > 0
        else:
            # FIXME: this should raise, as we don't have any non-service
            # workers, correct?
            pass

    # 4.1 Disable all CPU or GPU* workers.
    # (this should even be possible if no GPUs are available anyways)
    with Context.make_with('dask', gpus=0) as ctx:
        assert len(ctx.executor.get_available_workers().has_cpu()) > 0
        assert len(ctx.executor.get_available_workers().has_cuda()) == 0


class MockPartition:
    @property
    def slice(self):
        return Slice(origin=(0, 0, 0), shape=Shape((16, 16, 16), sig_dims=1))


class MockTask:
    def __init__(self, delay):
        self._delay = delay

    def __call__(self, params, env: Environment) -> Any:
        time.sleep(self._delay)
        return params

    def get_tracing_span_context(self):
        raise NotImplementedError()

    def get_partition(self):
        return MockPartition()

    def get_locations(self):
        return None

    def get_resources(self):
        return {}


class DelayingCommHandler(TaskCommHandler):
    def handle_task(self, task: TaskProtocol, queue: WorkerQueue):
        # our tests only work if the tasks don't all get submitted at the
        # beginning of the `run_tasks` call - this simulates the live
        # processing scenario
        time.sleep(0.25)


@pytest.mark.parametrize('executor', ['dask', 'pipelined', 'inline', 'concurrent'])
def test_scatter_update(executor, local_cluster_ctx, pipelined_ctx, concurrent_executor):
    import uuid
    cancel_id = str(uuid.uuid4())

    # we do this dance to re-use the existing executors, so we don't have to
    # mark this test as slow:
    if executor == 'dask':
        ctx = local_cluster_ctx
    elif executor == 'pipelined':
        ctx = pipelined_ctx
    elif executor == 'inline':
        ctx = Context.make_with('inline')
    elif executor == 'concurrent':
        ctx = Context(executor=concurrent_executor)
    else:
        raise ValueError('invalid executor name')

    print(f"starting for {executor}: {ctx.executor} {cancel_id}")

    exc = ctx.executor
    comm_handler = DelayingCommHandler()

    # for concurrent, we need to take the number of threads into account,
    # otherwise the tasks will start running too soon for the first update
    # to take place
    num_workers = sum(w.nthreads for w in exc.get_available_workers())

    results = []

    with exc.scatter('hello scatter') as handle:
        tasks = []
        # first two tasks: return immediately
        # (this ensures we get to the point of changing the params faster)
        tasks.append(MockTask(delay=0))
        tasks.append(MockTask(delay=0))
        # all that follow have some delay:
        tasks.extend([MockTask(delay=0.025) for _ in range(300 * num_workers)])
        result_iter = exc.run_tasks(
            cancel_id=cancel_id,
            params_handle=handle,
            task_comm_handler=comm_handler,
            tasks=tasks,
        )
        first_result, _task = next(result_iter)
        print("started")
        assert first_result == "hello scatter"
        print(first_result)
        exc.scatter_update(handle, 'new value')
        for result, _task in result_iter:
            results.append(result)
            # early exit to keep test as fast as possible:
            if result == "new value":
                result_iter.close()
                break

    # eventually, the new value is available to the workers:
    assert "new value" in results

    print("done")


@pytest.mark.parametrize('executor', ['dask', 'pipelined', 'inline', 'concurrent'])
def test_scatter_patch(executor, local_cluster_ctx, pipelined_ctx, concurrent_executor):
    import uuid
    cancel_id = str(uuid.uuid4())

    # we do this dance to re-use the existing executors, so we don't have to
    # mark this test as slow:
    if executor == 'dask':
        ctx = local_cluster_ctx
    elif executor == 'pipelined':
        ctx = pipelined_ctx
    elif executor == 'inline':
        ctx = Context.make_with('inline')
    elif executor == 'concurrent':
        ctx = Context(executor=concurrent_executor)
    else:
        raise ValueError('invalid executor name')

    print(f"starting for {executor}: {ctx.executor} {cancel_id}")

    exc = ctx.executor
    comm_handler = DelayingCommHandler()

    # for concurrent, we need to take the number of threads into account,
    # otherwise the tasks will start running too soon for the first update
    # to take place
    num_workers = sum(w.nthreads for w in exc.get_available_workers())

    results = []

    class Patchable:
        def __init__(self, value):
            self.value = value

        def patch(self, patch):
            self.value = patch

        def __repr__(self):
            return f"<Patchable: {self.value}>"

    with exc.scatter(Patchable('hello scatter')) as handle:
        tasks = []
        tasks.append(MockTask(delay=0))
        tasks.append(MockTask(delay=0))
        tasks.extend([MockTask(delay=0.025) for _ in range(300 * num_workers)])
        result_iter = exc.run_tasks(
            cancel_id=cancel_id,
            params_handle=handle,
            task_comm_handler=comm_handler,
            tasks=tasks,
        )
        first_result, _task = next(result_iter)
        print("started")
        assert first_result.value == "hello scatter"
        print(first_result)
        exc.scatter_update_patch(handle, 'new value')
        for result, _task in result_iter:
            results.append(result.value)
            if result.value == "new value":
                result_iter.close()
                break

    # eventually, the new value is available to the workers:
    assert "new value" in results

    print("done")
    print(results)
