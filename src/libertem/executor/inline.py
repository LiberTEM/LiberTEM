import cloudpickle
import psutil

from .base import JobExecutor, Environment
from .scheduler import Worker, WorkerSet
from libertem.common.backend import get_use_cuda


class InlineJobExecutor(JobExecutor):
    """
    naive JobExecutor that just iterates over partitions and processes them one after another
    """
    def __init__(self, debug=False, *args, **kwargs):
        self._debug = debug

    def run_tasks(self, tasks, cancel_id):
        threads = psutil.cpu_count(logical=False)
        env = Environment(threads_per_worker=threads)
        for task in tasks:
            if self._debug:
                cloudpickle.loads(cloudpickle.dumps(task))
            result = task(env=env)
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
            Worker(name='inline', host='localhost', resources=resources)
        ])
