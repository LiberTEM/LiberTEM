from .base import JobExecutor


class InlineJobExecutor(JobExecutor):
    def run_job(self, job):
        for task in job.get_tasks():
            yield task()
