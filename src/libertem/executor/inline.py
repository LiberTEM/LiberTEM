import cloudpickle
from .base import JobExecutor


class InlineJobExecutor(JobExecutor):
    """
    naive JobExecutor that just iterates over partitions and processes them one after another
    """
    def __init__(self, debug=False, *args, **kwargs):
        self._debug = debug

    def run_job(self, job, cancel_id=None):
        tasks = job.get_tasks()
        return self.run_tasks(tasks, cancel_id=job)

    def run_tasks(self, tasks, cancel_id):
        for task in tasks:
            if self._debug:
                cloudpickle.loads(cloudpickle.dumps(task))
            result = task()
            if self._debug:
                cloudpickle.loads(cloudpickle.dumps(result))
            yield result

    def run_function(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def get_available_workers(self):
        return dict(
            name='inline',
            host='localhost'
        )
