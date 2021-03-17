import numpy as np

from libertem.executor.inline import InlineJobExecutor
from libertem.udf import UDF


def test_run_each_worker_1():
    def fn1():
        return "some result"

    executor = InlineJobExecutor()

    results = executor.run_each_worker(fn1)
    assert len(results.keys()) == 1
    assert len(results.keys()) == len(executor.get_available_workers())

    k = next(iter(results))
    result0 = results[k]
    assert result0 == "some result"
    assert k == "inline"


class ThreadsPerWorkerUDF(UDF):
    def get_result_buffers(self):
        return {
            'num_threads': self.buffer(kind='nav', dtype=int),
        }

    def process_frame(self, frame):
        assert self.meta.threads_per_worker is not None,\
            "threads_per_worker should be an integer"
        self.results.num_threads[:] = self.meta.threads_per_worker


def test_inline_num_threads(lt_ctx, default_raw):
    threads = 2
    res = lt_ctx.run_udf(
        dataset=default_raw,
        udf=ThreadsPerWorkerUDF()
    )['num_threads']
    assert np.allclose(res, threads)
