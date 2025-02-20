import functools
import time
import random
from typing import Optional, TypeVar
from collections.abc import Generator
import multiprocessing as mp
import sys
import subprocess

import pytest
import numpy as np

from libertem.api import Context
from libertem.common.executor import (
    TaskCommHandler, TaskProtocol, WorkerQueue, JobCancelledError,
)
from libertem.udf.sum import SumUDF
from libertem.executor.pipelined import (
    PipelinedExecutor, WorkerPool, _order_results, pipelined_worker
)
import libertem.executor.pipelined
from libertem.udf import UDF
from libertem.common.exceptions import UDFRunCancelled
from libertem.io.dataset.memory import MemoryDataSet


class CustomException(Exception):
    pass


@pytest.fixture(scope="module")
def pipelined_ex():
    executor = None
    try:
        executor = PipelinedExecutor(
            spec=PipelinedExecutor.make_spec(cpus=range(2), cudas=[]),
            # to prevent issues in already-pinned situations (i.e. containerized
            # environments), don't pin our worker processes in testing:
            pin_workers=False,
            cleanup_timeout=5.,
        )
        yield executor
    finally:
        if executor is not None:
            print(f"pipelined_ex fixture: closing executor {id(executor)}")
            executor.close()


def test_pipelined_executor(pipelined_ex):
    executor = pipelined_ex
    ctx = Context(executor=executor)
    udf = SumUDF()
    data = np.random.randn(4, 4, 128, 128)
    ds = ctx.load("memory", data=data)
    res = ctx.run_udf(dataset=ds, udf=udf)
    assert np.allclose(
        data.sum(axis=(0, 1)),
        res['intensity'].data,
    )


def test_run_function(pipelined_ex):
    assert pipelined_ex.run_function(lambda: 42) == 42


class RaisesUDF(UDF):
    def __init__(self, exc_cls=CustomException):
        super().__init__(exc_cls=exc_cls)

    def get_result_buffers(self):
        return {
            "stuff": self.buffer(kind='nav'),
        }

    def process_frame(self, frame):
        raise self.params.exc_cls("what")


def test_udf_exception_queued(pipelined_ex):
    executor = pipelined_ex
    ctx = Context(executor=executor)

    data = np.random.randn(16, 16, 128, 128)
    ds = ctx.load("memory", data=data, num_partitions=16)

    error_udf = RaisesUDF()  # raises an error
    with pytest.raises(CustomException):  # raised by executor as expected
        ctx.run_udf(dataset=ds, udf=error_udf)

    normal_udf = SumUDF()
    ctx.run_udf(dataset=ds, udf=normal_udf)


@pytest.mark.slow
def test_default_spec():
    # make sure `.make_local` works:
    executor = None
    try:
        executor = PipelinedExecutor.make_local()

        # to at least see that something works:
        assert executor.run_function(lambda: 42) == 42
    finally:
        if executor is not None:
            executor.close()


def test_make_with():
    with Context.make_with("pipelined", cpus=1) as ctx:
        assert ctx.executor.run_function(lambda: 42) == 42


_STOP = object()

T = TypeVar('T')


def echo() -> Generator[Optional[T], T, None]:
    last = None
    while last is not _STOP:
        print(f"yielding {last}")
        last = yield last
        print(f"got {last}")
        if last is None:
            import traceback
            traceback.print_stack()


def test_order_results_in_order():
    r1 = object()
    t1 = object()
    t1id = 0

    r2 = object()
    t2 = object()
    t2id = 1

    r3 = object()
    t3 = object()
    t3id = 2

    # results come in as triples (result, task, task_id)
    def _trace_1():
        yield (r1, t1, t1id)
        yield (r2, t2, t2id)
        yield (r3, t3, t3id)

    # _order_results discards the task_id at the end
    ordered = _order_results(_trace_1())
    assert next(ordered) == (r1, t1)
    assert next(ordered) == (r2, t2)
    assert next(ordered) == (r3, t3)


def test_order_results_missing_task():
    r1 = object()
    t1 = object()
    t1id = 0

    r3 = object()
    t3 = object()
    t3id = 2

    # results come in as triples (result, task, task_id)
    def _trace_1():
        yield (r1, t1, t1id)
        yield (r3, t3, t3id)

    # _order_results discards the task_id at the end
    ordered = _order_results(_trace_1())
    assert next(ordered) == (r1, t1)
    with pytest.raises(RuntimeError):
        next(ordered)


def test_order_results_postponed_task():
    r1 = object()
    t1 = object()
    t1id = 0

    r2 = object()
    t2 = object()
    t2id = 1

    r3 = object()
    t3 = object()
    t3id = 2

    # results come in as triples (result, task, task_id)
    def _trace_1():
        yield (r1, t1, t1id)
        yield (r3, t3, t3id)
        yield (r2, t2, t2id)

    # _order_results discards the task_id at the end
    ordered = _order_results(_trace_1())
    assert next(ordered) == (r1, t1)
    assert next(ordered) == (r2, t2)
    assert next(ordered) == (r3, t3)


def test_run_function_failure(pipelined_ex):
    def _f():
        raise CustomException("this fails to run")

    with pytest.raises(CustomException) as ex_info:
        pipelined_ex.run_function(_f)

    assert ex_info.match("^this fails to run$")


def test_run_function_error():
    executor = None
    try:
        executor = PipelinedExecutor(
            spec=PipelinedExecutor.make_spec(cpus=range(2), cudas=[]),
            pin_workers=False,
        )

        def _break(a, b, c):
            raise CustomException("stuff is broken, can't do it.")

        def _do_patch_worker():
            # monkeypatch the pipelined module; this should only be run on
            # worker processes
            # XXX this completely destroys the workers ability to properly
            # function, but that's okay because it's a throwaway process pool
            # anyways:
            libertem.executor.pipelined.worker_run_function = _break

        executor.run_each_worker(_do_patch_worker)
        with pytest.raises(CustomException) as e:
            executor.run_function(lambda: 42)
        assert e.match("stuff is broken, can't do it.")
    finally:
        if executor is not None:
            executor.close()


def _broken_pipelined_worker(queues, pin, spec, span_context, early_setup):
    raise CustomException("stuff is broken, can't do it.")


def test_early_startup_error():
    """
    Simulate very early startup error, not even getting to the try/except
    that gives us error feedback via the queue.
    """
    executor = None

    # manual patching, we mock.patch doesn't work in multiprocessing
    # environments:
    original_pipelined_worker = libertem.executor.pipelined.pipelined_worker
    try:
        libertem.executor.pipelined.pipelined_worker = _broken_pipelined_worker
        with pytest.raises(RuntimeError) as e:
            executor = PipelinedExecutor(
                spec=PipelinedExecutor.make_spec(cpus=range(2), cudas=[]),
                pin_workers=False,
            )
        assert e.match("One or more workers failed to start")
    finally:
        libertem.executor.pipelined.pipelined_worker = original_pipelined_worker
        if executor is not None:
            executor.close()


def _patch_setup_device():
    def _broken_setup_device(spec, pin):
        """
        Broken version of pipelined._setup_device for error injection
        """
        raise RuntimeError("stuff is broken, can't do it.")
    libertem.executor.pipelined._setup_device = _broken_setup_device


def _slow():
    time.sleep(100)


def test_startup_error():
    """
    Simulate an error when starting up the worker, in this case we raise in
    _setup_device
    """
    executor = None
    try:
        with pytest.raises(RuntimeError) as e:
            executor = PipelinedExecutor(
                spec=PipelinedExecutor.make_spec(cpus=range(2), cudas=[]),
                pin_workers=False,
                early_setup=_patch_setup_device,
            )
        assert e.match("stuff is broken, can't do it.")
    finally:
        if executor is not None:
            executor.close()


def test_startup_timeout():
    executor = None
    try:
        with pytest.raises(RuntimeError) as e:
            executor = PipelinedExecutor(
                spec=PipelinedExecutor.make_spec(cpus=range(2), cudas=[]),
                pin_workers=False,
                early_setup=_slow,
                startup_timeout=0,
            )
        assert e.match("Timeout while starting workers")
    finally:
        if executor is not None:
            executor.close()


class FailEventuallyUDF(UDF):
    def get_result_buffers(self):
        return {
            "stuff": self.buffer(kind="nav"),
        }

    def process_partition(self, partition):
        if random.random() > 0.50:
            time.sleep(0.1)
        raise CustomException("stuff happens")


def test_failure_with_delay(pipelined_ex):
    ctx = Context(executor=pipelined_ex)
    udf = FailEventuallyUDF()
    data = np.random.randn(1, 32, 16, 16)
    ds = ctx.load("memory", data=data, num_partitions=32)
    with pytest.raises(CustomException) as e:
        ctx.run_udf(dataset=ds, udf=udf)
    assert e.match("stuff happens")


class SucceedEventuallyUDF(UDF):
    def get_result_buffers(self):
        return {
            "intensity": self.buffer(kind="nav"),
        }

    def process_partition(self, partition):
        if random.random() > 0.50:
            time.sleep(0.1)
        self.results.intensity[:] = np.sum(partition, axis=(-1, -2))


def test_success_with_delay(pipelined_ex):
    ctx = Context(executor=pipelined_ex)
    udf = SucceedEventuallyUDF()
    data = np.random.randn(1, 32, 16, 16)
    ds = ctx.load("memory", data=data, num_partitions=32)
    res = ctx.run_udf(dataset=ds, udf=udf)
    assert np.allclose(res['intensity'].data, np.sum(data, axis=(2, 3)))


def test_make_spec_multi_cuda():
    spec = PipelinedExecutor.make_spec(cpus=[0], cudas=[0, 1, 2, 2])
    assert spec == [
        {
            "device_id": 0,
            "name": "cuda-0-0",
            "device_kind": "CUDA",
            "worker_idx": 0,
            "has_cupy": False,
        },
        {
            "device_id": 1,
            "name": "cuda-1-0",
            "device_kind": "CUDA",
            "worker_idx": 1,
            "has_cupy": False,
        },
        {
            "device_id": 2,
            "name": "cuda-2-0",
            "device_kind": "CUDA",
            "worker_idx": 2,
            "has_cupy": False,
        },
        {
            "device_id": 2,
            "name": "cuda-2-1",
            "device_kind": "CUDA",
            "worker_idx": 3,
            "has_cupy": False,
        },
        {
            "device_id": 0,
            "name": "cpu-0",
            "device_kind": "CPU",
            "worker_idx": 4,
            "has_cupy": False,
        },
    ]


def test_make_spec_cpu_int():
    int_spec = PipelinedExecutor.make_spec(cpus=4, cudas=tuple(), has_cupy=True)
    range_spec = PipelinedExecutor.make_spec(cpus=range(4), cudas=tuple(), has_cupy=True)
    assert range_spec == int_spec


def test_make_spec_cuda_int():
    spec_n = 2
    cuda_spec = PipelinedExecutor.make_spec(cpus=[0], cudas=spec_n)
    num_cudas = 0
    for spec in cuda_spec:
        if spec['device_kind'] == 'CUDA':
            num_cudas += 1
    assert num_cudas == spec_n


def test_close_with_scatter():
    executor = None
    try:
        spec = PipelinedExecutor.make_spec(cpus=range(2), cudas=[])
        executor = PipelinedExecutor(spec=spec, pin_workers=False)
        with executor.scatter("huh"):
            executor.close()
    finally:
        if executor is not None:
            executor.close()


def test_exception_in_main_thread():
    executor = None
    try:
        spec = PipelinedExecutor.make_spec(cpus=range(2), cudas=[])
        executor = PipelinedExecutor(spec=spec, pin_workers=False)
        ctx = Context(executor=executor)
        udf = SucceedEventuallyUDF()
        data = np.random.randn(1, 32, 16, 16)
        ds = ctx.load("memory", data=data, num_partitions=32)
        res_iter = ctx.run_udf_iter(dataset=ds, udf=udf)
        with pytest.raises(RuntimeError):
            next(res_iter)
            res_iter.throw(RuntimeError("stuff"))
            next(res_iter)
        res_iter.close()

        # here, we get a KeyError if the worker queues aren't drained:
        print("second UDF run starting")
        ctx.run_udf(dataset=ds, udf=udf)
        print("second UDF run done")
    finally:
        if executor is not None:
            executor.close()


def test_kill_pool():
    spec = PipelinedExecutor.make_spec(cpus=range(2), cudas=[])
    pool = WorkerPool(
        worker_fn=functools.partial(
            pipelined_worker,
            pin=False,
        ),
        spec=spec,
    )
    # monkey-patch: disable terminate; force use of kill
    for worker_info in pool._workers:
        worker_info.process.terminate = lambda: None
    pool.kill(timeout=0.1)


def test_term_pool():
    spec = PipelinedExecutor.make_spec(cpus=range(2), cudas=[])
    pool = WorkerPool(
        worker_fn=functools.partial(
            pipelined_worker,
            pin=False,
        ),
        spec=spec,
    )
    # hopefully this will hit the `terminate` branch:
    pool.kill(timeout=5)


def _teardown_manual():
    subprocess.run([
        sys.executable,
        '-c',
        'import time;'
        'from libertem.api import Context;'
        'ctx=Context.make_with("pipelined", cpus=1);'
        'time.sleep(0.2);'
        'ctx.close()'
    ], timeout=30)


def test_auto_teardown_1():
    subprocess.run([
        sys.executable,
        '-c',
        'import time;'
        'from libertem.api import Context;'
        'ctx=Context.make_with("pipelined", cpus=1);'
    ], timeout=30)


def test_auto_teardown_2():
    subprocess.run([
        sys.executable,
        '-c',
        'import time;'
        'import gc;'
        'from libertem.api import Context;'
        'Context.make_with("pipelined", cpus=1);'
        'gc.collect();'
    ], timeout=30)


def test_manual_teardown():
    mp_ctx = mp.get_context("spawn")
    p = mp_ctx.Process(target=_teardown_manual, name="_teardown_manual")
    p.start()
    p.join(30)
    exitcode = p.exitcode
    if exitcode != 0:
        p.terminate()
        raise RuntimeError(f"exitcode is {exitcode}, should be 0")


class CancelledTaskCommHandler(TaskCommHandler):
    def handle_task(self, task: TaskProtocol, queue: WorkerQueue):
        raise JobCancelledError()

    def start(self):
        pass

    def done(self):
        pass


class CancelledMemoryDataSet(MemoryDataSet):
    def get_task_comm_handler(self) -> TaskCommHandler:
        return CancelledTaskCommHandler()


def test_cancellation(pipelined_ex, default_raw):
    executor = pipelined_ex
    ctx = Context(executor=executor)

    cancel_ds = CancelledMemoryDataSet(data=np.zeros((16, 16, 16, 16)))

    with pytest.raises(UDFRunCancelled) as ex:
        ctx.run_udf(dataset=cancel_ds, udf=SumUDF())

    assert ex.match(r"^UDF run cancelled after \d+ partitions$")

    # after cancellation, the executor is still usable:
    _ = ctx.run_udf(dataset=default_raw, udf=SumUDF())


def test_with_progress(pipelined_ex, default_raw):
    executor = pipelined_ex
    ctx = Context(executor=executor)

    udf = SumUDF()
    res_iter = ctx.run_udf_iter(dataset=default_raw, udf=udf, progress=True)
    try:
        for res in res_iter:
            pass
    finally:
        res_iter.close()
