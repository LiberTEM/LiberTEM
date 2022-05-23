import pytest
import numpy as np

from libertem.api import Context
from libertem.udf.sum import SumUDF
from libertem.executor.pipelined import PipelinedExecutor
# from libertem.io.dataset.memory import MemoryDataSet


@pytest.fixture
def pipelined_ex():
    try:
        executor = PipelinedExecutor(n_workers=2)
        yield executor
    finally:
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
