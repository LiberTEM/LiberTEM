from typing import Any, Iterable
import contextlib

from dask import delayed

from .base import JobExecutor, Environment, TaskProtocol
from .scheduler import Worker, WorkerSet


class DelayedJobExecutor(JobExecutor):
    """
    JobExecutor that uses dask.delayed to execute tasks.
    """
    @contextlib.contextmanager
    def scatter(self, obj):
        yield delayed(obj)

    def run_tasks(
        self,
        tasks: Iterable[TaskProtocol],
        params_handle: Any,
        cancel_id: Any,
    ):
        env = Environment(threads_per_worker=1)
        for task in tasks:
            result = delayed(task, nout=len(task._udf_classes))(env=env, params=params_handle)
            yield result, task

    def run_function(self, fn, *args, **kwargs):
        result = fn(*args, **kwargs)
        return result

    def run_wrap(self, fn, *args, **kwargs):
        result = delayed(fn)(*args, **kwargs)
        return result

    def map(self, fn, iterable):
        d_fn = delayed(fn)
        return [d_fn(item)
                for item in iterable]

    def run_each_host(self, fn, *args, **kwargs):
        return {"localhost": fn(*args, **kwargs)}

    def run_each_worker(self, fn, *args, **kwargs):
        return {"delayed": fn(*args, **kwargs)}

    def get_available_workers(self):
        resources = {"compute": 1, "CPU": 1}

        return WorkerSet([
            Worker(name='delayed', host='localhost', resources=resources)
        ])
