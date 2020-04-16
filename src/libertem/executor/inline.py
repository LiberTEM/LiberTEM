import cloudpickle
from .base import JobExecutor
from .scheduler import Worker, WorkerSet


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
        return WorkerSet([
            Worker(name='inline', host='localhost')
        ])
