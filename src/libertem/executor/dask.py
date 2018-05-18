from dask import distributed as dd
from .base import JobExecutor


class DaskJobExecutor(JobExecutor):
    def __init__(self, scheduler_uri):
        self.scheduler_uri = scheduler_uri
        self.client = dd.Client(self.scheduler_uri, processes=False)

    def run_job(self, job):
        futures = [
            self.client.submit(task, workers=task.get_locations())
            for task in job.get_tasks()
        ]
        for future, result in dd.as_completed(futures, with_results=True):
            yield result
