from typing import Any, Iterable
import cloudpickle
import psutil
import contextlib

from .base import JobExecutor, Environment, TaskProtocol
from .scheduler import Worker, WorkerSet
from libertem.common.backend import get_use_cuda


class InlineJobExecutor(JobExecutor):
    """
    Naive JobExecutor that just iterates over partitions and processes them one after another

    Parameters
    ----------
    debug : bool
        Set this to enable additional serializability checks

    inline_threads : Optional[int]
        How many fine grained threads should be allowed? Leaving this `None` will
        allow one thread per CPU core
    """
    def __init__(self, debug=False, inline_threads=None, *args, **kwargs):
        # Only import if actually instantiated, i.e. will likely be used
        import libertem.preload  # noqa: 401
        self._debug = debug
        self._inline_threads = inline_threads

    @contextlib.contextmanager
    def scatter(self, obj):
        yield obj

    def run_tasks(
        self,
        tasks: Iterable[TaskProtocol],
        params_handle: Any,
        cancel_id: Any,
    ):
        threads = self._inline_threads
        if threads is None:
            threads = psutil.cpu_count(logical=False)
        env = Environment(threads_per_worker=threads, threaded_executor=False)
        for task in tasks:
            if self._debug:
                cloudpickle.loads(cloudpickle.dumps(task))
            result = task(env=env, params=params_handle)
            if self._debug:
                cloudpickle.loads(cloudpickle.dumps(result))
            yield result, task

    def run_function(self, fn, *args, **kwargs):
        if self._debug:
            cloudpickle.loads(cloudpickle.dumps((fn, args, kwargs)))
        result = fn(*args, **kwargs)
        if self._debug:
            cloudpickle.loads(cloudpickle.dumps(result))
        return result

    def map(self, fn, iterable):
        return [fn(item)
                for item in iterable]

    def run_each_host(self, fn, *args, **kwargs):
        if self._debug:
            cloudpickle.loads(cloudpickle.dumps((fn, args, kwargs)))
        return {"localhost": fn(*args, **kwargs)}

    def run_each_worker(self, fn, *args, **kwargs):
        return {"inline": fn(*args, **kwargs)}

    def get_available_workers(self):
        resources = {"compute": 1, "CPU": 1}
        if get_use_cuda() is not None:
            resources["CUDA"] = 1

        return WorkerSet([
            Worker(name='inline', host='localhost', resources=resources, nthreads=1)
        ])
