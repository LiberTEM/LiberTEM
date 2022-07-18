from typing import Generator, Optional, TypeVar

import pytest
import numpy as np

from libertem.api import Context
from libertem.udf.sum import SumUDF
from libertem.executor.pipelined import PipelinedExecutor, _order_results
from libertem.udf import UDF


@pytest.fixture
def pipelined_ex():
    executor = None
    try:
        executor = PipelinedExecutor(
            spec=PipelinedExecutor.make_spec(cpus=range(2), cudas=[]),
            # to prevent issues in already-pinned situations (i.e. containerized
            # environments), don't pin our worker processes in testing:
            pin_workers=False,
        )
        yield executor
    finally:
        if executor is not None:
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
    def __init__(self, exc_cls=Exception):
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
    with pytest.raises(RuntimeError):  # raised by executor as expected
        ctx.run_udf(dataset=ds, udf=error_udf)

    # Fails immediately on run_udf because queue is in bad state
    normal_udf = SumUDF()
    ctx.run_udf(dataset=ds, udf=normal_udf)
    # Error is raised during the task dispatch loop when we check if any tasks
    # completed yet


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
        raise Exception("this fails to run")

    with pytest.raises(RuntimeError) as ex_info:
        pipelined_ex.run_function(_f)

    assert ex_info.match("^failed to run function: this fails to run$")


def test_worker_loop_error():
    executor = None
    try:
        executor = PipelinedExecutor(
            spec=PipelinedExecutor.make_spec(cpus=range(2), cudas=[]),
            pin_workers=False,
        )

        def _break(a, b, c):
            raise RuntimeError("stuff is broken, can't do it.")

        def _do_patch_worker():
            # XXX this completely destroys the workers ability to properly
            # function, but that's okay because it's a throwaway process pool
            # anyways:
            from libertem.executor import pipelined
            pipelined.worker_run_function = _break

        executor.run_function(_do_patch_worker)
        with pytest.raises(RuntimeError) as e:
            executor.run_function(lambda: 42)
        assert e.match("failed to run function: stuff is broken, can't do it.")
    finally:
        if executor is not None:
            executor.close()
