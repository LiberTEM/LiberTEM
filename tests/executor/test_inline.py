import numpy as np
import pytest

from libertem.executor.inline import InlineJobExecutor
from libertem.udf import UDF
from libertem.udf.sum import SumUDF
from libertem.common.executor import (
    TaskCommHandler, TaskProtocol, WorkerQueue, JobCancelledError,
)
from libertem.common.exceptions import UDFRunCancelled
from libertem.io.dataset.memory import MemoryDataSet


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
        assert self.meta.threads_per_worker is not None, \
            "threads_per_worker should be an integer"
        self.results.num_threads[:] = self.meta.threads_per_worker


def test_inline_num_threads(lt_ctx, default_raw):
    threads = 2
    res = lt_ctx.run_udf(
        dataset=default_raw,
        udf=ThreadsPerWorkerUDF()
    )['num_threads']
    assert np.allclose(res, threads)


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


def test_cancellation(lt_ctx, default_raw):
    cancel_ds = CancelledMemoryDataSet(data=np.zeros((16, 16, 16, 16)))

    with pytest.raises(UDFRunCancelled) as ex:
        lt_ctx.run_udf(dataset=cancel_ds, udf=SumUDF())

    assert ex.match(r"^UDF run cancelled after \d+ partitions$")

    # after cancellation, the executor is still usable:
    _ = lt_ctx.run_udf(dataset=default_raw, udf=SumUDF())
