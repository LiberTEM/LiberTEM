import logging
from copy import deepcopy

from dask import delayed

from .scheduler import Worker, WorkerSet
from .base import JobExecutor


log = logging.getLogger(__name__)


class InlineDaskJobExecutor(JobExecutor):
    """
    JobExecutor that uses dask.delayed to run tasks in parallel.
    """
    def _validate_resources(self, resources):
        if 'CUDA' in resources:
            raise RuntimeError(
                "Requesting CUDA resource on a cluster without resource management."
            )

    def _get_delayeds(self, tasks):
        delayeds = []
        for task in tasks:
            self._validate_resources(task.get_resources())
            delayeds.append(
                # Deepcopy to make sure we don't accidentally share resources or
                # create side effects between tasks and are thread safe
                delayed(deepcopy(task))()
            )
        return delayeds

    def run_tasks(self, tasks, cancel_id):
        tasks = list(tasks)
        delayeds = self._get_delayeds(tasks)
        results = delayed()(delayeds).compute()
        for result, task in zip(results, tasks):
            yield result, task

    def run_function(self, fn, *args, **kwargs):
        return delayed(fn)(*args, **kwargs).compute()

    def map(self, fn, iterable):
        delayeds = [delayed(fn)(item) for item in iterable]
        return delayed()(delayeds).compute()

    def run_each_host(self, fn, *args, **kwargs):
        return {"localhost": delayed(fn)(*args, **kwargs).compute()}

    def run_each_worker(self, fn, *args, **kwargs):
        raise NotImplementedError()

    def get_available_workers(self):
        resources = {"compute": 1, "CPU": 1}

        return WorkerSet([
            Worker(name='inlinedask', host='localhost', resources=resources)
        ])
