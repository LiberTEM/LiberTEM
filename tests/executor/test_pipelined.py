import pytest
import numpy as np

from libertem.api import Context
from libertem.udf.sum import SumUDF
from libertem.executor.pipelined import PipelinedExecutor
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
