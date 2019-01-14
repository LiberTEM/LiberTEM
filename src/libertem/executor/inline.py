import functools

from .base import JobExecutor


class InlineJobExecutor(JobExecutor):
    """
    naive JobExecutor that just iterates over partitions and processes them one after another
    """
    def run_job(self, job):
        for task in job.get_tasks():
            yield task()

    def run_function(self, fn, *args, **kwargs):
        return fn(*args, **kwargs)

    def map_partitions(self, dataset, fn, *args, **kwargs):
        for partition in dataset.get_partitions():
            fn_kwargs = {}
            fn_kwargs.update(kwargs)
            fn_kwargs.update({'partition': partition})
            fn_bound = functools.partial(fn, *args, **fn_kwargs)
            yield fn_bound()
