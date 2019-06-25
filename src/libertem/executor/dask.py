import functools
import logging
import signal

import tornado.util
from dask import distributed as dd

from .base import JobExecutor, JobCancelledError, sync_to_async, AsyncAdapter


log = logging.getLogger(__name__)


class CommonDaskMixin(object):
    def _task_idx_to_workers(self, workers, idx):
        hosts = list(sorted(set(w['host'] for w in workers)))
        host_idx = idx % len(hosts)
        host = hosts[host_idx]
        return [
            w['name']
            for w in workers
            if w['host'] == host
        ]

    def _get_futures(self, tasks):
        futures = []
        available_workers = self.get_available_workers()
        if len(available_workers) == 0:
            raise RuntimeError("no workers available!")
        for task in tasks:
            submit_kwargs = {}
            locations = task.get_locations()
            if locations is not None and len(locations) == 0:
                raise ValueError("no workers found for task")
            if locations is None:
                locations = self._task_idx_to_workers(available_workers, task.idx)
            submit_kwargs['workers'] = locations
            futures.append(
                self.client.submit(task, **submit_kwargs)
            )
        return futures

    def get_available_workers(self):
        info = self.client.scheduler_info()
        return [
            {
                'name': worker['name'],
                'host': worker['host'],
            }
            for worker in info['workers'].values()
        ]


class DaskJobExecutor(CommonDaskMixin, JobExecutor):
    def __init__(self, client, is_local=False):
        self.is_local = is_local
        self.client = client
        self._futures = {}

    def run_job(self, job, cancel_id=None):
        tasks = job.get_tasks()
        return self.run_tasks(tasks, cancel_id=cancel_id)

    def run_tasks(self, tasks, cancel_id):
        futures = self._get_futures(tasks)
        self._futures[cancel_id] = futures
        try:
            for future, result in dd.as_completed(futures, with_results=True):
                if future.cancelled():
                    del self._futures[cancel_id]
                    raise JobCancelledError()
                yield result
        finally:
            if cancel_id in self._futures:
                del self._futures[cancel_id]

    def cancel(self, cancel_id):
        if cancel_id in self._futures:
            futures = self._futures[cancel_id]
            self.client.cancel(futures)

    def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn`
        """
        fn_with_args = functools.partial(fn, *args, **kwargs)
        future = self.client.submit(fn_with_args, priority=1)
        return future.result()

    def close(self):
        if self.is_local:
            if self.client.cluster is not None:
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
        return cls(client=client, is_local=False, *args, **kwargs)

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

        # Disable handling Ctrl-C on the workers for a local cluster
        # since the nanny restarts workers in that case and that gets mixed
        # with Ctrl-C handling of the main process, at least on Windows
        client.run(functools.partial(signal.signal, signal.SIGINT, signal.SIG_IGN))

        return cls(client=client, is_local=True)


class AsyncDaskJobExecutor(AsyncAdapter):
    def __init__(self, wrapped=None, *args, **kwargs):
        if wrapped is None:
            wrapped = DaskJobExecutor(*args, **kwargs)
        super().__init__(wrapped)

    @classmethod
    async def connect(cls, scheduler_uri, *args, **kwargs):
        executor = await sync_to_async(functools.partial(
            DaskJobExecutor.connect,
            scheduler_uri=scheduler_uri,
            *args,
            **kwargs,
        ))
        return cls(wrapped=executor)

    @classmethod
    async def make_local(cls, cluster_kwargs=None, client_kwargs=None):
        executor = await sync_to_async(functools.partial(
            DaskJobExecutor.make_local,
            cluster_kwargs=cluster_kwargs,
            client_kwargs=client_kwargs,
        ))
        return cls(wrapped=executor)
