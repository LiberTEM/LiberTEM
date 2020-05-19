import os
from unittest import mock

import cloudpickle
from .base import JobExecutor
from .scheduler import Worker, WorkerSet
from libertem.udf.backend import get_use_cuda


class InlineJobExecutor(JobExecutor):
    """
    naive JobExecutor that just iterates over partitions and processes them one after another
    """
    def __init__(self, debug=False, *args, **kwargs):
        self._debug = debug

    def run_job(self, job, cancel_id=None):
        tasks = job.get_tasks()
        for result, task in self.run_tasks(tasks, cancel_id=job):
            yield result

    def run_tasks(self, tasks, cancel_id):
        # The UDFRunner expects one of these environment variable to be set
        if (os.environ.get("LIBERTEM_USE_CPU") or os.environ.get("LIBERTEM_USE_CUDA")):
            patch = {}
        else:
            patch = {'LIBERTEM_USE_CPU': "0"}
        with mock.patch.dict(os.environ, patch):
            for task in tasks:
                if self._debug:
                    cloudpickle.loads(cloudpickle.dumps(task))
                result = task()
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

    def get_available_workers(self):
        resources = {"compute": 1, "CPU": 1}
        if get_use_cuda() is not None:
            resources["CUDA"] = 1

        return WorkerSet([
            Worker(name='inline', host='localhost', resources=resources)
        ])
