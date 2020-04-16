import functools
import logging
import signal

import tornado.util
from dask import distributed as dd

from .base import JobExecutor, JobCancelledError, sync_to_async, AsyncAdapter
from .scheduler import Worker, WorkerSet


log = logging.getLogger(__name__)


class TaskProxy:
    def __init__(self, task, task_id):
        self.task = task
        self.task_id = task_id

    def __getattr__(self, k):
        if k in ["task"]:
            return super().__getattr__(k)
        return getattr(self.task, k)

    def __call__(self, *args, **kwargs):
        return {
            "task_result": self.task(),
            "task_id": self.task_id,
        }


class CommonDaskMixin(object):
    def _task_idx_to_workers(self, workers, idx):
        hosts = list(sorted(workers.hosts()))
        host_idx = idx % len(hosts)
        host = hosts[host_idx]
        return workers.filter(lambda w: w.host == host)

    def _futures_for_locations(self, fns_and_locations):
        """
        Submit tasks and return the resulting futures

        Parameters
        ----------

        fns_and_locations : List[Tuple[callable,WorkerSet]]
            callables zipped with potential locations
        """
        futures = []
        for task, locations in fns_and_locations:
            submit_kwargs = {}
            if locations is not None:
                if len(locations) == 0:
                    raise ValueError("no workers found for task")
                locations = locations.names()
            submit_kwargs['workers'] = locations
            futures.append(
                self.client.submit(task, **submit_kwargs)
            )
        return futures

    def _get_futures(self, tasks):
        available_workers = self.get_available_workers()
        if len(available_workers) == 0:
            raise RuntimeError("no workers available!")
        return self._futures_for_locations([
            (
                task,
                task.get_locations() or self._task_idx_to_workers(
                    available_workers, task.idx)
            )
            for task in tasks
        ])

    def get_available_workers(self):
        info = self.client.scheduler_info()
        return WorkerSet([
            Worker(name=worker['name'], host=worker['host'])
            for worker in info['workers'].values()
        ])


class DaskJobExecutor(CommonDaskMixin, JobExecutor):
    def __init__(self, client, is_local=False):
        self.is_local = is_local
        self.client = client
        self._futures = {}

    def run_job(self, job, cancel_id=None):
        tasks = job.get_tasks()
        for result, task in self.run_tasks(tasks, cancel_id=cancel_id):
            yield result

    def run_tasks(self, tasks, cancel_id):
        tasks = list(tasks)
        tasks_wrapped = []

        def _id_to_task(task_id):
            return tasks[task_id]

        for idx, orig_task in enumerate(tasks):
            tasks_wrapped.append(TaskProxy(orig_task, idx))

        futures = self._get_futures(tasks_wrapped)
        self._futures[cancel_id] = futures

        try:
            for future, result_wrap in dd.as_completed(futures, with_results=True):
                if future.cancelled():
                    del self._futures[cancel_id]
                    raise JobCancelledError()
                result = result_wrap['task_result']
                task = _id_to_task(result_wrap['task_id'])
                yield result, task
        finally:
            if cancel_id in self._futures:
                del self._futures[cancel_id]

    def cancel(self, cancel_id):
        if cancel_id in self._futures:
            futures = self._futures[cancel_id]
            self.client.cancel(futures)

    def run_each_partition(self, partitions, fn, all_nodes=False):
        """
        Run `fn` for all partitions. Yields results in order of completion.

        Parameters
        ----------

        partitions : List[Partition]
            List of relevant partitions.

        fn : callable
            Function to call, will get the partition as first and only argument.

        all_nodes : bool
            If all_nodes is True, run the function on all nodes that have this partition,
            otherwise run on any node that has the partition. If a partition has no location,
            the function will not be run for that partition if `all_nodes` is True, otherwise
            it will be run on any node.
        """
        def _make_items_all():
            for p in partitions:
                locs = p.get_locations()
                if locs is None:
                    continue
                for workers in locs.group_by_host():
                    yield (lambda: fn(p), workers)

        if all_nodes:
            items = _make_items_all()
        else:
            items = ((lambda: fn(p), p.get_locations())
                     for p in partitions)
        futures = self._futures_for_locations(items)
        # TODO: do we need cancellation and all that good stuff?
        for future, result in dd.as_completed(futures, with_results=True):
            if future.cancelled():
                raise JobCancelledError()
            yield result

    def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn` on any worker
        """
        fn_with_args = functools.partial(fn, *args, **kwargs)
        future = self.client.submit(fn_with_args, priority=1)
        return future.result()

    def map(self, fn, iterable):
        """
        Run a callable `fn` for each element in `iterable`, on arbitrary worker nodes.

        Parameters
        ----------

        fn : callable
            Function to call. Should accept exactly one parameter.

        iterable : Iterable
            Which elements to call the function on.
        """
        return [future.result()
                for future in self.client.map(fn, iterable)]

    def run_each_host(self, fn, *args, **kwargs):
        """
        Run a callable `fn` once on each host, gathering all results into a dict host -> result

        TODO: any cancellation/errors to handle?
        """
        available_workers = self.get_available_workers()

        future_map = {}
        for worker_set in available_workers.group_by_host():
            future_map[worker_set.example().host] = self.client.submit(
                functools.partial(fn, *args, **kwargs),
                priority=1,
                workers=worker_set.names(),
                # NOTE: need pure=False, otherwise the functions will all map to the same
                # scheduler key and will only run once
                pure=False,
            )
        result_map = {
            host: future.result()
            for host, future in future_map.items()
        }
        return result_map

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
