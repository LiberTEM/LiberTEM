import tornado.util
from dask import distributed as dd
from distributed.asyncio import AioClient
from .base import JobExecutor, AsyncJobExecutor, JobCancelledError


# NOTE:
# if you are mistakenly using dd.Client in an asyncio environment,
# you get a message like this:
# error message: "RuntimeError: Non-thread-safe operation invoked on an event loop
# other than the current one"
# related: debugging via env var PYTHONASYNCIODEBUG=1


class CommonDaskMixin(object):
    def _get_futures(self, job):
        futures = []
        for task in job.get_tasks():
            submit_kwargs = {}
            if not self.is_local:
                submit_kwargs['workers'] = task.get_locations()
            futures.append(
                self.client.submit(task, **submit_kwargs)
            )
        return futures


class AsyncDaskJobExecutor(CommonDaskMixin, AsyncJobExecutor):
    def __init__(self, client, is_local=False):
        self.is_local = is_local
        self.client = client
        self._futures = {}

    async def close(self):
        await self.client.close()
        if self.is_local:
            try:
                self.client.cluster.close(timeout=1)
            except tornado.util.TimeoutError:
                pass

    async def run_job(self, job):
        futures = self._get_futures(job)
        self._futures[job] = futures
        async for future, result in dd.as_completed(futures, with_results=True):
            if future.cancelled():
                raise JobCancelledError()
            yield result
        del self._futures[job]

    async def cancel_job(self, job):
        if job in self._futures:
            futures = self._futures[job]
            await self.client.cancel(futures)

    @classmethod
    async def connect(cls, scheduler_uri, *args, **kwargs):
        """
        Connect to remote dask scheduler

        Returns
        -------
        AsyncDaskJobExecutor
            the connected JobExecutor
        """
        client = await AioClient(address=scheduler_uri)
        return cls(client=client, *args, **kwargs)

    @classmethod
    async def make_local(cls, cluster_kwargs=None, client_kwargs=None):
        """
        Spin up a local dask cluster

        interesting cluster_kwargs:
            threads_per_worker
            n_workers

        Returns
        -------
        AsyncDaskJobExecutor
            the connected JobExecutor
        """
        cluster = dd.LocalCluster(**(cluster_kwargs or {}))
        client = await AioClient(cluster, **(client_kwargs or {}))
        return cls(client=client, is_local=True)


class DaskJobExecutor(CommonDaskMixin, JobExecutor):
    def __init__(self, client, is_local=False):
        self.is_local = is_local
        self.client = client

    def run_job(self, job):
        futures = self._get_futures(job)
        for future, result in dd.as_completed(futures, with_results=True):
            yield result

    def close(self):
        if self.is_local:
            try:
                self.client.cluster.close(timeout=1)
            except tornado.util.TimeoutError:
                pass
        self.client.close()

    @classmethod
    def connect(cls, scheduler_uri, *args, **kwargs):
        """
        Connect to a remote dask scheduler

        Returns
        -------
        DaskJobExecutor
            the connected JobExecutor
        """
        client = dd.Client(address=scheduler_uri)
        return cls(client=client, *args, **kwargs)

    @classmethod
    def make_local(cls, cluster_kwargs=None, client_kwargs=None):
        """
        Spin up a local dask cluster

        interesting cluster_kwargs:
            threads_per_worker
            n_workers

        Returns
        -------
        DaskJobExecutor
            the connected JobExecutor
        """
        cluster = dd.LocalCluster(**(cluster_kwargs or {}))
        client = dd.Client(cluster, **(client_kwargs or {}))
        return cls(client=client, is_local=True)
