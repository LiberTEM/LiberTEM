from dask import distributed as dd
from .base import JobExecutor


class DaskJobExecutor(JobExecutor):
    def __init__(self, scheduler_uri=None, client=None, is_local=False):
        self.is_local = is_local
        if client is not None:
            if scheduler_uri:
                raise ValueError("pass either client or scheduler_uri, not both")
            self.client = client
        else:
            if client:
                raise ValueError("pass either client or scheduler_uri, not both")
            self.client = dd.Client(scheduler_uri, processes=False)

    def close(self):
        # FIXME: need to cleanup anything else? cluster?
        self.client.close()

    @classmethod
    def make_local(cls, cluster_kwargs=None, client_kwargs=None):
        """
        interesting cluster_kwargs:
            threads_per_worker
            n_workers
        """
        cluster = dd.LocalCluster(**(cluster_kwargs or {}))
        client = dd.Client(cluster, **(client_kwargs or {}))
        return cls(client=client, is_local=True)

    def run_job(self, job):
        futures = []
        for task in job.get_tasks():
            submit_kwargs = {}
            if not self.is_local:
                submit_kwargs['workers'] = task.get_locations()
            futures.append(
                self.client.submit(task, **submit_kwargs)
            )
        for future, result in dd.as_completed(futures, with_results=True):
            yield result
