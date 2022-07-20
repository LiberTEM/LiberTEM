import contextlib
from copy import deepcopy
import functools
import logging
import signal
from typing import Iterable, Any, Optional, Tuple

from dask import distributed as dd
import dask

from libertem.common.threading import set_num_threads_env

from .base import BaseJobExecutor, AsyncAdapter
from libertem.common.executor import (
    JobCancelledError, TaskCommHandler, TaskProtocol, Environment,
)
from libertem.common.async_utils import sync_to_async
from libertem.common.scheduler import Worker, WorkerSet
from libertem.common.backend import set_use_cpu, set_use_cuda
from libertem.common.async_utils import adjust_event_loop_policy


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


def cluster_spec(
        cpus: Iterable[int], cudas: Iterable[int], has_cupy: bool,
        name: str = 'default', num_service: int = 1, options: Optional[dict] = None,
        preload: Optional[Tuple[str]] = None):
    '''
    Create a worker specification dictionary for a LiberTEM Dask cluster

    The return from this function can be passed to :code:`DaskJobExecutor.make_local(spec=spec)`.

    This creates a Dask cluster spec with special initializations and resource tags
    for CPU + GPU processing in LiberTEM.
    See :ref:`cluster spec` for an example.
    See http://distributed.dask.org/en/stable/api.html#distributed.SpecCluster
    for more info on cluster specs.

    Parameters
    ----------
    cpus
        IDs for CPU workers. Currently no pinning is used, i.e. this specifies the total
        number and identification of workers, not the CPU cores that are used.
    cudas
        IDs for CUDA device workers. LiberTEM will use the IDs specified here. This
        has to match CUDA device IDs on the system. Specify the same ID multiple times
        to spawn multiple workers on the same CUDA device.
    has_cupy
        Specify if the cluster should signal that it supports GPU-based array programming using
        CuPy
    name
        Prefix for the worker names
    num_service
        Number of additional workers that are reserved for service tasks. Computation tasks
        will not be scheduled on these workers, which guarantees responsive behavior for file
        browsing etc.
    options
        Options to pass through to every worker. See Dask documentation for details
    preload
        Items to preload on workers in addition to LiberTEM-internal preloads.
        This can be used to load libraries, for example HDF5 filter plugins before h5py is used.
        See https://docs.dask.org/en/stable/how-to/customize-initialization.html#preload-scripts
        for more information.

    See also
    --------
    :func:`libertem.utils.devices.detect`
    '''

    if options is None:
        options = {}
    if preload is None:
        preload = ()
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

    def _get_tracing_setup(service_name: str, service_id: str) -> str:
        return (
            f"from libertem.common.tracing import maybe_setup_tracing; "
            f"maybe_setup_tracing(service_name='{service_name}', service_id='{service_id}')"
        )

    for cpu in cpus:
        worker_name = f'{name}-cpu-{cpu}'
        cpu_spec = deepcopy(cpu_base_spec)
        cpu_spec['options']['preload'] = preload + (
            'from libertem.executor.dask import worker_setup; '
            + f'worker_setup(resource="CPU", device={cpu})',
            _get_tracing_setup(worker_name, str(cpu)),
            'libertem.preload',
        )
        workers_spec[worker_name] = cpu_spec

    for service in range(num_service):
        worker_name = f'{name}-service-{service}'
        service_spec = deepcopy(service_base_spec)
        service_spec['options']['preload'] = preload + (
            _get_tracing_setup(worker_name, str(service)),
            'libertem.preload',
        )
        workers_spec[worker_name] = service_spec

    for cuda in cudas:
        worker_name = f'{name}-cuda-{cuda}'
        if worker_name in workers_spec:
            num_with_name = sum(n.startswith(worker_name) for n in workers_spec)
            worker_name = f'{worker_name}-{num_with_name - 1}'
        cuda_spec = deepcopy(cuda_base_spec)
        cuda_spec['options']['preload'] = preload + (
            'from libertem.executor.dask import worker_setup; '
            + f'worker_setup(resource="CUDA", device={cuda})',
            _get_tracing_setup(worker_name, str(cuda)),
            'libertem.preload',
        )
        workers_spec[worker_name] = cuda_spec

    return workers_spec


def _run_task(task, params, task_id, threaded_executor):
    """
    Very simple wrapper function. As dask internally caches functions that are
    submitted to the cluster in various ways, we need to make sure to
    consistently use the same function, and not build one on the fly.

    Without this function, UDFTask->UDF->UDFData ends up in the
    cache, which blows up memory usage over time.
    """
    env = Environment(threads_per_worker=1, threaded_executor=threaded_executor)
    task_result = task(env=env, params=params)
    return {
        "task_result": task_result,
        "task_id": task_id,
    }


def _simple_run_task(task):
    return task()


class CommonDaskMixin:
    def _task_idx_to_workers(self, workers, idx):
        hosts = list(sorted(workers.hosts()))
        host_idx = idx % len(hosts)
        host = hosts[host_idx]
        return workers.filter(lambda w: w.host == host)

    def _future_for_location(
        self, task, locations, resources, workers, task_args=None, wrap_fn=_run_task,
    ):
        """
        Submit tasks and return the resulting futures

        Parameters
        ----------
        task:
            callable
        locations:
            potential locations to run the task
        resources:
            required resources for the task
        workers : WorkerSet
            Available workers in the cluster
        """
        submit_kwargs = {}
        if task_args is None:
            task_args = {}
        if locations is not None:
            if len(locations) == 0:
                raise ValueError("no workers found for task")
            locations = locations.names()
        submit_kwargs.update({
            'resources': self._validate_resources(workers, resources),
            'workers': locations,
            'pure': False,
        })
        return self.client.submit(
            wrap_fn, task, *task_args, **submit_kwargs
        )

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

    def _get_future(self, task, workers, idx, params_handle, threaded_executor):
        if len(workers) == 0:
            raise RuntimeError("no workers available!")
        return self._future_for_location(
            task,
            task.get_locations() or self._task_idx_to_workers(
                workers, idx
            ),
            task.get_resources(),
            workers,
            task_args=(
                params_handle,
                idx,
                threaded_executor
            )
        )

    def get_available_workers(self) -> WorkerSet:
        info = self.client.scheduler_info()
        return WorkerSet([
            Worker(
                name=worker['name'],
                host=worker['host'],
                resources=worker['resources'],
                nthreads=worker['nthreads'],
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


class DaskJobExecutor(CommonDaskMixin, BaseJobExecutor):
    '''
    Default LiberTEM executor that uses `Dask futures
    <https://docs.dask.org/en/stable/futures.html>`_.

    Parameters
    ----------

    client : distributed.Client
    is_local : bool
        Close the Client and cluster when the executor is closed.
    lt_resources : bool
        Specify if the cluster has LiberTEM resource tags and environment
        variables for GPU processing. Autodetected by default.
    '''
    def __init__(self, client: dd.Client, is_local: bool = False,
                lt_resources: bool = None):
        self.is_local = is_local
        self.client = client
        if lt_resources is None:
            lt_resources = self.has_libertem_resources()
        self.lt_resources = lt_resources
        self._futures = {}

    @contextlib.contextmanager
    def scatter(self, obj):
        yield self.client.scatter(obj, broadcast=True)

    def run_tasks(
        self,
        tasks: Iterable[TaskProtocol],
        params_handle: Any,
        cancel_id: Any,
        task_comm_handler: TaskCommHandler,
    ):
        tasks = list(tasks)
        tasks_w_index = list(enumerate(tasks))

        def _id_to_task(task_id):
            return tasks[task_id]

        workers = self.get_available_workers()
        threaded_executor = workers.has_threaded_workers()

        self._futures[cancel_id] = []
        initial = []

        for w in range(int(len(workers))):
            if not tasks_w_index:
                break
            idx, wrapped_task = tasks_w_index.pop(0)
            future = self._get_future(wrapped_task, workers, idx, params_handle, threaded_executor)
            initial.append(future)
            self._futures[cancel_id].append(future)

        try:
            as_completed = dd.as_completed(initial, with_results=True, loop=self.client.loop)
            for future, result_wrap in as_completed:
                if future.cancelled():
                    del self._futures[cancel_id]
                    raise JobCancelledError()
                result = result_wrap['task_result']
                task = _id_to_task(result_wrap['task_id'])
                if tasks_w_index:
                    idx, wrapped_task = tasks_w_index.pop(0)
                    future = self._get_future(
                        wrapped_task, workers, idx, params_handle, threaded_executor,
                    )
                    as_completed.add(future)
                    self._futures[cancel_id].append(future)
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
        workers = self.get_available_workers()
        futures = [
            self._future_for_location(*item, workers, wrap_fn=_simple_run_task)
            for item in items
        ]
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
        """
        # TODO: any cancellation/errors to handle?
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
        # Client.run() creates issues on Windows and OS X with Python 3.6
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
    def connect(cls, scheduler_uri, *args, client_kwargs: Optional[dict] = None, **kwargs):
        """
        Connect to a remote dask scheduler.

        Parameters
        ----------
        scheduler_uri: str
            Compatible with the :code:`address` parameter of :class:`distributed.Client`.
        client_kwargs: dict or None
            Passed as kwargs to :class:`distributed.Client`.
            :code:`client_kwargs['set_as_default']` is set to :code:`False`
            unless specified otherwise to avoid interference with Dask-based workflows.
            Pass :code:`client_kwargs={'set_as_default': True}` to set the Client as the
            default Dask scheduler and keep it running when the Context closes.
        *args, **kwargs: Passed to :class:`DaskJobExecutor`.

        Returns
        -------
        DaskJobExecutor
            the connected JobExecutor
        """
        if client_kwargs is None:
            client_kwargs = {}
        if client_kwargs.get('set_as_default') is None:
            client_kwargs['set_as_default'] = False
        is_local = not client_kwargs['set_as_default']
        client = dd.Client(address=scheduler_uri, **client_kwargs)
        return cls(client=client, is_local=is_local, *args, **kwargs)

    @classmethod
    def make_local(cls, spec: Optional[dict] = None, cluster_kwargs: Optional[dict] = None,
            client_kwargs: Optional[dict] = None, preload: Optional[Tuple[str]] = None):
        """
        Spin up a local dask cluster

        Parameters
        ----------
        spec
            Dask cluster spec, see
            http://distributed.dask.org/en/stable/api.html#distributed.SpecCluster
            for more info.
            :func:`libertem.utils.devices.detect` allows to detect devices that can be used
            with LiberTEM, and :func:`cluster_spec` can be used to create a :code:`spec`
            with customized parameters.
        cluster_kwargs
            Passed to :class:`distributed.SpecCluster`.
        client_kwargs
            Passed to :class:`distributed.Client`. Pass
            :code:`client_kwargs={'set_as_default': True}` to set the Client as the
            default Dask scheduler.
        preload: Optional[Tuple[str]]
            Passed to :func:`cluster_spec` if :code:`spec` is :code:`None`.

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
            spec = cluster_spec(**detect(), preload=preload)
        else:
            if preload is not None:
                raise ValueError(
                    "Passing both spec and preload is not supported. "
                    "Instead, include preloading specification in the spec"
                )
        if client_kwargs is None:
            client_kwargs = {}
        if client_kwargs.get('set_as_default') is None:
            client_kwargs['set_as_default'] = False

        if cluster_kwargs is None:
            cluster_kwargs = {}
        if cluster_kwargs.get('silence_logs') is None:
            cluster_kwargs['silence_logs'] = logging.WARN

        with set_num_threads_env(n=1):
            # Mitigation for https://github.com/dask/distributed/issues/6776
            with dask.config.set({"distributed.worker.profile.enabled": False}):
                cluster = dd.SpecCluster(workers=spec, **(cluster_kwargs or {}))
                client = dd.Client(cluster, **(client_kwargs or {}))
                client.wait_for_workers(len(spec))

        is_local = not client_kwargs['set_as_default']

        return cls(client=client, is_local=is_local, lt_resources=True)

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


def cli_worker(
        scheduler, local_directory, cpus, cudas, has_cupy, name, log_level, preload: Tuple[str]):
    import asyncio

    options = {
        "silence_logs": log_level,
        "local_directory": local_directory

    }

    spec = cluster_spec(
        cpus=cpus, cudas=cudas, has_cupy=has_cupy, name=name, options=options, preload=preload)

    async def run(spec):
        # Mitigation for https://github.com/dask/distributed/issues/6776
        with dask.config.set({"distributed.worker.profile.enabled": False}):
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
