import pytest

from libertem.api import Context
from libertem.udf.base import NoOpUDF


@pytest.fixture(scope='module')
def mod_ctx():
    """
    To make it easy to experiment with different executors
    and their parameters, we have a local fixture here.

    Note: make sure not to use `lt_ctx`, as that has some debugging
    enabled by default.
    """
    yield Context.make_with('inline')


class NoOpBufUDF(NoOpUDF):
    """Like NoOpUDF, but with a result buffer"""

    def get_result_buffers(self):
        return {
            # a bit larger buffer, so that its overheads can be measured
            # (for the test dataset, a single kind=sig buffer has barely any overhead)
            "sigbuf": self.buffer(
                kind="sig", dtype=int, where="device", extra_shape=(100,)
            ),
        }

    def merge(self, dest, src):
        pass


@pytest.mark.benchmark(
    group="udf overheads"
)
def test_noop_udf(mod_ctx, benchmark, medium_raw, set_affinity):
    """
    This measures running a no-op UDF. The overheads measured include:

    - Main-node preparations
    - Scheduling and running of tasks
    - Reading of all data (which might be just returning slices from a memory map)
    """
    benchmark(
        mod_ctx.run_udf,
        dataset=medium_raw,
        udf=NoOpUDF()
    )


@pytest.mark.benchmark(
    group="udf overheads"
)
def test_noop_udf_sig(mod_ctx, benchmark, medium_raw, set_affinity):
    """
    This measures running a no-op UDF with a result buffer.

    In addition to `test_noop_udf`, this also includes allocation
    of buffers.

    """
    benchmark(
        mod_ctx.run_udf,
        dataset=medium_raw,
        udf=NoOpBufUDF()
    )
