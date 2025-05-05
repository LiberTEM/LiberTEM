import os
import sys
import re
import subprocess

import numpy as np
import numba
import pytest
import psutil

from libertem.executor.dask import (
    CommonDaskMixin
)
from libertem.common.scheduler import Worker, WorkerSet
from libertem.executor.dask import cluster_spec, DaskJobExecutor
from libertem.executor.inline import InlineJobExecutor
from libertem.udf.base import UDFRunner
from libertem.udf.base import NoOpUDF
from libertem.udf.sum import SumUDF
from libertem.udf.raw import PickUDF
from libertem.io.dataset.memory import MemoryDataSet
from libertem.api import Context
from libertem.udf import UDF

from utils import _mk_random


def test_task_affinity_1():
    cdm = CommonDaskMixin()

    ws1 = WorkerSet([
        Worker(host='127.0.0.1', name='w1', resources={}, nthreads=1),
        Worker(host='127.0.0.1', name='w2', resources={}, nthreads=1),
        Worker(host='127.0.0.1', name='w3', resources={}, nthreads=1),
        Worker(host='127.0.0.1', name='w4', resources={}, nthreads=1),
    ])
    ws2 = WorkerSet([
        Worker(host='127.0.0.2', name='w5', resources={}, nthreads=1),
        Worker(host='127.0.0.2', name='w6', resources={}, nthreads=1),
        Worker(host='127.0.0.2', name='w7', resources={}, nthreads=1),
        Worker(host='127.0.0.2', name='w8', resources={}, nthreads=1),
    ])
    workers = ws1.extend(ws2)

    assert cdm._task_idx_to_workers(workers, 0) == ws1
    assert cdm._task_idx_to_workers(workers, 1) == ws2
    assert cdm._task_idx_to_workers(workers, 2) == ws1
    assert cdm._task_idx_to_workers(workers, 3) == ws2


@pytest.mark.skipif(os.name == 'nt',
                    reason="Doesn't run on windows")
@pytest.mark.asyncio
async def test_fd_limit(async_executor):
    import resource

    # set soft limit, throws errors but allows to raise it
    # again afterwards:
    proc = psutil.Process()
    oldlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (proc.num_fds() + 24, oldlimit[1]))

    print("fds", proc.num_fds())

    try:
        data = _mk_random(size=(1, 16, 16), dtype='<u2')
        dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16), num_partitions=1)

        roi = np.ones((1,), dtype=bool)
        udf = PickUDF()

        for i in range(32):
            print(i)
            print(proc.num_fds())

            async for part in UDFRunner([udf]).run_for_dataset_async(
                dataset=dataset,
                executor=async_executor,
                cancel_id="42",
                roi=roi,
            ):
                pass
    finally:
        resource.setrlimit(resource.RLIMIT_NOFILE, oldlimit)


def test_run_each_partition(dask_executor):
    data = _mk_random(size=(16, 16, 16), dtype='<u2')
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16), num_partitions=16)
    partitions = dataset.get_partitions()

    def fn1(partition):
        return 42

    i = 0
    for result in dask_executor.run_each_partition(partitions, fn1, all_nodes=False):
        i += 1
        assert result == 42
    assert i == 16


def test_run_each_partition_2(dask_executor):
    data = _mk_random(size=(16, 16, 16), dtype='<u2')
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16), num_partitions=16)
    partitions = dataset.get_partitions()

    i = 0
    for result in dask_executor.run_each_partition(partitions, lambda p: False, all_nodes=True):
        i += 1
    assert i == 0  # memory dataset doesn't have a defined location, so fn is never run


def test_map_1(dask_executor):
    iterable = [1, 2, 3]
    res = dask_executor.map(lambda x: x**2, iterable)
    assert res == [1, 4, 9]


def test_multiple_clients(local_cluster_url, default_raw):
    ex1 = DaskJobExecutor.connect(local_cluster_url)

    # this creates a second Client, and even though we are setting `set_as_default=False`,
    # this Client is then used by functions like `dd.as_completed`. That is because
    # `set_as_default` only sets the dask scheduler config to "dask.distributed", it does
    # not affect setting the _client_ as the global default `Client`!
    # so any time `as_completed` is called, the `loop` needs to be set correctly, otherwise
    # this may result in strange hangs and crashes
    ex2 = DaskJobExecutor.connect(local_cluster_url)

    udf = SumUDF()

    cx1 = Context(executor=ex1)
    cx1.run_udf(dataset=default_raw, udf=udf)

    ex1.client.close()
    ex2.client.close()


def test_run_each_worker_1(dask_executor):
    def fn1():
        return "some result"
    results = dask_executor.run_each_worker(fn1)
    assert len(results.keys()) >= 1
    assert len(results.keys()) == len(dask_executor.get_available_workers())
    k = next(iter(results))
    result0 = results[k]
    assert result0 == "some result"


@numba.njit(parallel=True)
def something(x):
    result = 0
    for i in numba.prange(23):
        result += x
    return result


class ThreadsPerWorkerUDF(UDF):
    def get_result_buffers(self):
        return {
            'num_threads': self.buffer(kind='nav', dtype=int),
        }

    # Just run once per partition
    def postprocess(self):
        # Actually do Numba multithreading to start threads and
        # freeze maximum thread count
        something(1)

    def process_frame(self, frame):
        assert self.meta.threads_per_worker is not None, \
            "threads_per_worker should be an integer"
        self.results.num_threads[:] = self.meta.threads_per_worker


def test_threads_per_worker(default_raw, dask_executor):
    ctx = Context(executor=dask_executor)
    inline_ctx = Context(executor=InlineJobExecutor())
    res = ctx.run_udf(dataset=default_raw, udf=ThreadsPerWorkerUDF())['num_threads']
    res_inline = inline_ctx.run_udf(dataset=default_raw, udf=ThreadsPerWorkerUDF())['num_threads']
    assert np.allclose(res, 1)
    assert np.allclose(res_inline, psutil.cpu_count(logical=False))


@pytest.mark.slow
def test_threads_per_worker_vanilla(default_raw, monkeypatch):
    old_threads = os.environ.get('NUMBA_NUM_THREADS')
    # Triggers #1053
    monkeypatch.delenv('NUMBA_NUM_THREADS', raising=False)
    with Context() as ctx:
        assert 'NUMBA_NUM_THREADS' not in os.environ
        # We have to reset it properly since it is set in pytest.ini
        # and Numba will complain if it is changed
        if old_threads:
            os.environ['NUMBA_NUM_THREADS'] = old_threads
        inline_ctx = Context(executor=InlineJobExecutor())
        res = ctx.run_udf(dataset=default_raw, udf=ThreadsPerWorkerUDF())
        res_inline = inline_ctx.run_udf(dataset=default_raw, udf=ThreadsPerWorkerUDF())
        print(res['num_threads'].data)
        assert np.all(res['num_threads'].data == 1)
        print(res_inline['num_threads'].data)
        assert np.all(res_inline['num_threads'].data == psutil.cpu_count(logical=False))


@pytest.mark.slow
@pytest.mark.skipif(
    'DASK_DISTRIBUTED__LOGGING__DISTRIBUTED' in os.environ,
    reason='Setting this value could influence test result',
)
def test_distributed_log_info():
    # Check if there are any distributed.<> - INFO
    # messages in the startup logs
    # Our default config is log_level == WARN
    result = subprocess.run(
        [
            sys.executable,
            '-c',
            'from libertem.api import Context;'
            'ctx = Context.make_with("dask", cpus=1);'
        ],
        timeout=30,
        capture_output=True,
        text=True,
    )

    assert re.search(r'distributed.\S+ - INFO', result.stderr) is None


def test_num_cores(local_cluster_ctx: Context):
    ctx = local_cluster_ctx
    num_cores_ds = ctx.load('memory', data=np.zeros((2, 3, 4, 5)))
    workers = ctx.executor.get_available_workers()
    cpu_count = len(workers.has_cpu())
    gpu_count = len(workers.has_cuda())
    assert num_cores_ds._cores == max(cpu_count, gpu_count)


def test_cluster_spec_cpu_int():
    int_spec = cluster_spec(cpus=4, cudas=tuple(), has_cupy=True)
    range_spec = cluster_spec(cpus=range(4), cudas=tuple(), has_cupy=True)
    assert range_spec == int_spec


def test_cluster_spec_cudas_int():
    spec_n = 4
    cuda_spec = cluster_spec(cpus=tuple(), cudas=spec_n, has_cupy=True)
    num_cudas = 0
    for spec in cuda_spec.values():
        num_cudas += spec.get('options', {}).get('resources', {}).get('CUDA', 0)
    assert num_cudas == spec_n


@pytest.mark.slow
def test_preload(hdf5_ds_1):
    # We don't use all since that might be too many
    cpus = (0, 1)
    hdf5_ds_1.set_num_cores(len(cpus))

    class CheckEnvUDF(NoOpUDF):
        def process_tile(self, tile):
            assert os.environ['LT_TEST_1'] == 'hello'
            assert os.environ['LT_TEST_2'] == 'world'

    preloads = (
        "import os; os.environ['LT_TEST_1'] = 'hello'",
        "import os; os.environ['LT_TEST_2'] = 'world'",
    )

    spec = cluster_spec(cpus=cpus, cudas=(), has_cupy=False, preload=preloads)
    with DaskJobExecutor.make_local(spec=spec) as executor:
        ctx = Context(executor=executor)
        ctx.run_udf(udf=CheckEnvUDF(), dataset=hdf5_ds_1)


def test_connected_cluster_cannot_snooze(local_cluster_url):
    """
    Normally if we go via .connect there is no way to
    setup a snooze_manager, and directly calling _scale_up
    and _scale_down is unsupported, but check anyway
    """
    ex = DaskJobExecutor.connect(local_cluster_url)
    assert ex.is_local
    assert ex.client.cluster is None
    num_workers = len(ex.get_available_workers())
    ex._scale_down()  # should be a no-op
    assert num_workers == len(ex.get_available_workers())
    ex._scale_up()  # should be a no-op
    assert num_workers == len(ex.get_available_workers())


@pytest.mark.slow
def test_local_cluster_snooze():
    try:
        ctx = Context.make_with('dask', cpus=2, gpus=0, snooze_timeout=10_000)
        num_workers = len(ctx.executor.get_available_workers())
        assert num_workers == 2 + 1  # +service worker
        ctx.executor.snooze_manager.snooze()
        assert len(ctx.executor.get_available_workers()) == 1
        ctx.executor.snooze_manager.unsnooze()
        assert len(ctx.executor.get_available_workers()) == 2 + 1
    finally:
        ctx.close()


@pytest.mark.slow
def test_scatter_keeps_alive():
    try:
        ctx = Context.make_with('dask', cpus=1, gpus=0, snooze_timeout=10_000)
        assert ctx.executor.snooze_manager.keep_alive == 0
        with ctx.executor.scatter("Hello!") as _:
            assert ctx.executor.snooze_manager.keep_alive == 1
        assert ctx.executor.snooze_manager.keep_alive == 0
    finally:
        ctx.close()
