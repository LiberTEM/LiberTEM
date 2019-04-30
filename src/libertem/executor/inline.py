from .base import JobExecutor


class InlineJobExecutor(JobExecutor):
    """
    naive JobExecutor that just iterates over partitions and processes them one after another
    """
    def run_job(self, job):
        tasks = job.get_tasks()
        return self.run_tasks(tasks, cancel_id=job)

    def run_tasks(self, tasks, cancel_id):
        for task in tasks:
            yield task()

    def run_function(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def get_available_workers(self):
        return dict(
            name='inline',
            host='localhost'
        )
