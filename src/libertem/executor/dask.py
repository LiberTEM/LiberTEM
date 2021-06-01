from copy import deepcopy
import functools
import logging
import signal

from dask import distributed as dd

from libertem.utils.threading import set_num_threads_env

from .base import (
    JobExecutor, JobCancelledError, sync_to_async, AsyncAdapter, TaskProxy,
    Environment,
)
from .scheduler import Worker, WorkerSet
from libertem.common.backend import set_use_cpu, set_use_cuda
from libertem.utils.async_utils import adjust_event_loop_policy


log = logging.getLogger(__name__)


def worker_setup(resource, device):
    # Disable handling Ctrl-C on the workers for a local cluster
    # since the nanny restarts workers in that case and that gets mixed
    # with Ctrl-C handling of the main process, at least on Windows
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    if resource == "CUDA":
        set_use_cuda(device)
    elif resource == "CPU":
        set_use_cpu(device)
    else:
        raise ValueError("Unknown resource %s, use 'CUDA' or 'CPU'", resource)


def cluster_spec(cpus, cudas, has_cupy, name='default', num_service=1, options=None):

    if options is None:
        options = {}
    if options.get("nthreads") is None:
        options["nthreads"] = 1
    if options.get("silence_logs") is None:
        options["silence_logs"] = logging.WARN

    workers_spec = {}

    cpu_options = deepcopy(options)
    cpu_options["resources"] = {"CPU": 1, "compute": 1, "ndarray": 1}
    cpu_base_spec = {
        "cls": dd.Nanny,
        "options": cpu_options,
    }

    # Service workers not for computation

    service_options = deepcopy(options)
    service_options["resources"] = {}
    service_base_spec = {
        "cls": dd.Nanny,
        "options": service_options
    }

    cuda_options = deepcopy(options)
    cuda_options["resources"] = {"CUDA": 1, "compute": 1}
    if has_cupy:
        cuda_options["resources"]["ndarray"] = 1
    cuda_base_spec = {
        "cls": dd.Nanny,
        "options": cuda_options
    }

    for cpu in cpus:
        cpu_spec = deepcopy(cpu_base_spec)
        cpu_spec['options']['preload'] = \
            'from libertem.executor.dask import worker_setup; ' + \
            f'worker_setup(resource="CPU", device={cpu})'
        workers_spec[f'{name}-cpu-{cpu}'] = cpu_spec

    for service in range(num_service):
        workers_spec[f'{name}-service-{service}'] = deepcopy(service_base_spec)

    for cuda in cudas:
        cuda_spec = deepcopy(cuda_base_spec)
        cuda_spec['options']['preload'] = \
            'from libertem.executor.dask import worker_setup; ' + \
            f'worker_setup(resource="CUDA", device={cuda})'
        workers_spec[f'{name}-cuda-{cuda}'] = cuda_spec

    return workers_spec


class DaskTaskProxy(TaskProxy):
    def __init__(self, task, task_id):
        super().__init__(task)
        self.task_id = task_id

    def __call__(self, *args, **kwargs):
        env = Environment(threads_per_worker=1)
        task_result = self.task(env=env)
        return {
            "task_result": task_result,
            "task_id": self.task_id,
        }

    def __repr__(self):
        return "<TaskProxy: %r (id=%s)>" % (self.task, self.task_id)


class CommonDaskMixin(object):
    def _task_idx_to_workers(self, workers, idx):
        hosts = list(sorted(workers.hosts()))
        host_idx = idx % len(hosts)
        host = hosts[host_idx]
        return workers.filter(lambda w: w.host == host)

    def _futures_for_locations(self, fns_and_meta):
        """
        Submit tasks and return the resulting futures

        Parameters
        ----------

        fns_and_meta : List[Tuple[callable,WorkerSet, dict]]
            callables zipped with potential locations and required resources.
        """
        workers = self.get_available_workers()
        futures = []
        for task, locations, resources in fns_and_meta:
            submit_kwargs = {}
            if locations is not None:
                if len(locations) == 0:
                    raise ValueError("no workers found for task")
                locations = locations.names()
            submit_kwargs.update({
                'resources': self._validate_resources(workers, resources),
                'workers': locations,
                'pure': False,
            })
            futures.append(
                self.client.submit(task, **submit_kwargs)
            )
        return futures

    def _validate_resources(self, workers, resources):
        # This is set in the constructor of DaskJobExecutor
        if self.lt_resources:
            if not self._resources_available(workers, resources):
                raise RuntimeError("Requested resources not available in cluster:", resources)
            result = resources
        else:
            if 'CUDA' in resources:
                raise RuntimeError(
                    "Requesting CUDA resource on a cluster without resource management."
                )
            result = {}
        return result

    def _resources_available(self, workers, resources):
        def filter_fn(worker):
            return all(worker.resources.get(key, 0) >= resources[key] for key in resources.keys())

        return len(workers.filter(filter_fn))

    def has_libertem_resources(self):
        workers = self.get_available_workers()

        def has_resources(worker):
            r = worker.resources
            return 'compute' in r and (('CPU' in r and 'ndarray' in r) or 'CUDA' in r)

        return len(workers.filter(has_resources)) > 0

    def _get_futures(self, tasks):
        available_workers = self.get_available_workers()
        if len(available_workers) == 0:
            raise RuntimeError("no workers available!")
        return self._futures_for_locations([
            (
                task,
                task.get_locations() or self._task_idx_to_workers(
                    available_workers, task.idx),
                task.get_resources()
            )
            for task in tasks
        ])

    def get_available_workers(self):
        info = self.client.scheduler_info()
        return WorkerSet([
            Worker(
                name=worker['name'],
                host=worker['host'],
                resources=worker['resources']
            )
            for worker in info['workers'].values()
        ])

    def get_resource_details(self):
        workers = self.get_available_workers()
        details = {}

        for worker in workers:
            host_name = worker.host
            if worker.name.startswith("tcp"):
                # for handling `dask-worker`
                # `dask-worker` name starts with "tcp"
                #  only supports CPU
                resource = 'cpu'
            else:
                # for handling `libertem-worker`
                r = worker.resources
                if "CPU" in r:
                    resource = 'cpu'
                elif "CUDA" in r:
                    resource = 'cuda'
                else:
                    resource = 'service'

            if host_name not in details.keys():
                details[host_name] = {
                                 'host': host_name,
                                 'cpu': 0,
                                 'cuda': 0,
                                 'service': 0,
                            }
            details[host_name][resource] += 1

        details_sorted = []
        for host in sorted(details.keys()):
            details_sorted.append(details[host])

        return details_sorted


class DaskJobExecutor(CommonDaskMixin, JobExecutor):
    def __init__(self, client, is_local=False, lt_resources=None):
        self.is_local = is_local
        self.client = client
        if lt_resources is None:
            lt_resources = self.has_libertem_resources()
        self.lt_resources = lt_resources
        self._futures = {}

    def run_tasks(self, tasks, cancel_id):
        tasks = list(tasks)
        tasks_wrapped = []

        def _id_to_task(task_id):
            return tasks[task_id]

        for idx, orig_task in enumerate(tasks):
            tasks_wrapped.append(DaskTaskProxy(orig_task, idx))

        futures = self._get_futures(tasks_wrapped)
        self._futures[cancel_id] = futures

        try:
            as_completed = dd.as_completed(futures, with_results=True, loop=self.client.loop)
            for future, result_wrap in as_completed:
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
            # TODO check if we should request a compute resource
            items = ((lambda: fn(p), p.get_locations(), {})
                     for p in partitions)
        futures = self._futures_for_locations(items)
        # TODO: do we need cancellation and all that good stuff?
        for future, result in dd.as_completed(futures, with_results=True, loop=self.client.loop):
            if future.cancelled():
                raise JobCancelledError()
            yield result

    def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn` on any worker
        """
        fn_with_args = functools.partial(fn, *args, **kwargs)
        future = self.client.submit(fn_with_args, priority=1, pure=False)
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
                for future in self.client.map(fn, iterable, pure=False)]

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

    def run_each_worker(self, fn, *args, **kwargs):
        available_workers = self.get_available_workers()

        future_map = {}
        for worker in available_workers:
            future_map[worker.name] = self.client.submit(
                functools.partial(fn, *args, **kwargs),
                priority=1,
                workers=[worker.name],
                # NOTE: need pure=False, otherwise the functions will all map to the same
                # scheduler key and will only run once
                pure=False,
            )
        result_map = {
            name: future.result()
            for name, future in future_map.items()
        }
        return result_map

    def close(self):
        if self.is_local:
            if self.client.cluster is not None:
                self.client.cluster.close(timeout=30)
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
        client = dd.Client(address=scheduler_uri, set_as_default=False)
        return cls(client=client, is_local=False, *args, **kwargs)

    @classmethod
    def make_local(cls, spec=None, cluster_kwargs=None, client_kwargs=None):
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

        # Distributed doesn't adjust the event loop policy when being run
        # from within pytest as of version 2.21.0. For that reason we
        # adjust the policy ourselves here.
        adjust_event_loop_policy()

        if spec is None:
            from libertem.utils.devices import detect
            spec = cluster_spec(**detect())
        if client_kwargs is None:
            client_kwargs = {}
        if client_kwargs.get('set_as_default') is None:
            client_kwargs['set_as_default'] = False

        if cluster_kwargs is None:
            cluster_kwargs = {}
        if cluster_kwargs.get('silence_logs') is None:
            cluster_kwargs['silence_logs'] = logging.WARN

        with set_num_threads_env(n=1):
            cluster = dd.SpecCluster(workers=spec, **(cluster_kwargs or {}))
            client = dd.Client(cluster, **(client_kwargs or {}))
            client.wait_for_workers(len(spec))

        return cls(client=client, is_local=True, lt_resources=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


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
    async def make_local(cls, spec=None, cluster_kwargs=None, client_kwargs=None):
        executor = await sync_to_async(functools.partial(
            DaskJobExecutor.make_local,
            spec=spec,
            cluster_kwargs=cluster_kwargs,
            client_kwargs=client_kwargs,
        ))
        return cls(wrapped=executor)


def cli_worker(scheduler, local_directory, cpus, cudas, has_cupy, name, log_level):
    import asyncio

    options = {
        "silence_logs": log_level,
        "local_directory": local_directory

    }

    spec = cluster_spec(cpus=cpus, cudas=cudas, has_cupy=has_cupy, name=name, options=options)

    async def run(spec):
        workers = []
        for name, spec in spec.items():
            cls = spec['cls']
            workers.append(
                cls(scheduler, name=name, **spec['options'])
            )
        import asyncio
        await asyncio.gather(*workers)
        for w in workers:
            await w.finished()

    asyncio.get_event_loop().run_until_complete(run(spec))
