import os

import numpy as np
import pytest

from libertem.executor.dask import (
    CommonDaskMixin
)
from libertem.common import Shape, Slice
from libertem.executor.scheduler import Worker, WorkerSet
from libertem.executor.dask import DaskJobExecutor
from libertem.job.raw import PickFrameJob
from libertem.udf.sum import SumUDF
from libertem.io.dataset.memory import MemoryDataSet
from libertem.api import Context

from utils import _mk_random


def test_task_affinity_1():
    cdm = CommonDaskMixin()

    ws1 = WorkerSet([
        Worker(host='127.0.0.1', name='w1', resources={}),
        Worker(host='127.0.0.1', name='w2', resources={}),
        Worker(host='127.0.0.1', name='w3', resources={}),
        Worker(host='127.0.0.1', name='w4', resources={}),
    ])
    ws2 = WorkerSet([
        Worker(host='127.0.0.2', name='w5', resources={}),
        Worker(host='127.0.0.2', name='w6', resources={}),
        Worker(host='127.0.0.2', name='w7', resources={}),
        Worker(host='127.0.0.2', name='w8', resources={}),
    ])
    workers = ws1.extend(ws2)

    assert cdm._task_idx_to_workers(workers, 0) == ws1
    assert cdm._task_idx_to_workers(workers, 1) == ws2
    assert cdm._task_idx_to_workers(workers, 2) == ws1
    assert cdm._task_idx_to_workers(workers, 3) == ws2


@pytest.mark.asyncio
async def test_run_job(async_executor):
    data = _mk_random(size=(16, 16, 16, 16), dtype='<u2')
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16), num_partitions=2)
    expected = data[0, 0]

    slice_ = Slice(origin=(0, 0, 0), shape=Shape((1, 16, 16), sig_dims=2))
    job = PickFrameJob(dataset=dataset, slice_=slice_)
    out = job.get_result_buffer()

    async for tiles in async_executor.run_job(job, cancel_id="42"):
        for tile in tiles:
            tile.reduce_into_result(out)

    assert out.shape == (1, 16, 16)
    assert np.allclose(out, expected)


@pytest.mark.skipif(os.name == 'nt',
                    reason="Doesn't run on windows")
@pytest.mark.asyncio
async def test_fd_limit(async_executor):
    import resource
    import psutil
    # set soft limit, throws errors but allows to raise it
    # again afterwards:
    proc = psutil.Process()
    oldlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (proc.num_fds() + 24, oldlimit[1]))

    print("fds", proc.num_fds())

    try:
        data = _mk_random(size=(1, 16, 16), dtype='<u2')
        dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16), num_partitions=1)

        slice_ = Slice(origin=(0, 0, 0), shape=Shape((1, 16, 16), sig_dims=2))
        job = PickFrameJob(dataset=dataset, slice_=slice_)

        for i in range(32):
            print(i)
            print(proc.num_fds())
            async for tiles in async_executor.run_job(job, cancel_id="42"):
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
    DaskJobExecutor.connect(local_cluster_url)

    udf = SumUDF()

    cx1 = Context(executor=ex1)
    cx1.run_udf(dataset=default_raw, udf=udf)
