from .base import JobExecutor


class InlineJobExecutor(JobExecutor):
    """
    naive JobExecutor that just iterates over partitions and processes them one after another
    """
    def run_job(self, job):
        for task in job.get_tasks():
            yield task()
