import os
import contextlib
from copy import deepcopy
import functools
import logging
import copy
import time
import signal
from typing import Any, Optional, Union, Callable
from collections.abc import Iterable
import uuid

from dask import distributed as dd
import dask

from libertem.common.threading import set_num_threads_env

from .base import BaseJobExecutor, AsyncAdapter, ResourceError
from libertem.common.executor import (
    JobCancelledError, TaskCommHandler, TaskProtocol, Environment, WorkerContext,
)
from libertem.common.snooze import SnoozeManager, keep_alive, keep_alive_context
from libertem.common.subscriptions import SubscriptionManager
from libertem.common.async_utils import sync_to_async
from libertem.common.scheduler import Worker, WorkerSet
from libertem.common.backend import set_use_cpu, set_use_cuda
from libertem.common.async_utils import adjust_event_loop_policy
from .utils import assign_cudas

log = logging.getLogger(__name__)


class DaskWorkerContext(WorkerContext):
    def __init__(self, comms_topic: Optional[str]):
        # DaskWorkerContext sends all messages via a single unique topic id
        # which are later unpacked on the client node; this allows us to handle
        # concurrent runs using the same executor with separate comms channels
        self._comms_topic = comms_topic

    @property
    def dask_worker(self):
        try:
            return self._worker
        except AttributeError:
            self._worker = dd.get_worker()
            return self._worker

    def signal(self, ident: str, topic: str, msg_dict: dict[str, Any]):
        if self._comms_topic is None:
            # Scheduler Dask does not have comms so don't send
            return
        msg_dict.update({'ident': ident, 'topic': topic})
        try:
            self.dask_worker.log_event(self._comms_topic, msg_dict)
        except AttributeError:
            # No structured logs available in this Dask
            # Catch the exception here just in case there is
            # version / API mismatch
            pass


@contextlib.contextmanager
def set_worker_log_level(level: Union[str, int], force: bool = False):
    """
    Set the dask.distributed log level for any processes spawned
    within the context manager. If force is False, don't overwrite
    any existing environment variable.
    """
    env_keys = ['DASK_DISTRIBUTED__LOGGING__DISTRIBUTED']
    try:
        old_env = {
            k: os.environ[k]
            for k in env_keys
            if k in os.environ
        }
        os.environ.update({
            k: str(level)
            for k in env_keys
            if force or (k not in old_env)
        })
        yield
    finally:
        os.environ.update(old_env)
        for key in env_keys:
            if key not in old_env:
                del os.environ[key]


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
    cpus: Union[int, Iterable[int]],
    cudas: Union[int, Iterable[int]],
    has_cupy: bool,
    name: str = 'default',
    num_service: int = 1,
    options: Optional[dict] = None,
    preload: Optional[tuple[str, ...]] = None
):
    '''
    Create a worker specification dictionary for a LiberTEM Dask cluster

    The return from this function can be passed to :code:`DaskJobExecutor.make_local(spec=spec)`.

    This creates a Dask cluster spec with special initializations and resource tags
    for CPU + GPU processing in LiberTEM.
    See :ref:`cluster spec` for an example.
    See https://distributed.dask.org/en/stable/api.html#distributed.SpecCluster
    for more info on cluster specs.

    Parameters
    ----------
    cpus: int | Iterable[int]
        IDs for CPU workers as an iterable, or an integer number of workers to create.
        Currently no pinning is used, i.e. this specifies the total
        number and identification of workers, not the CPU cores that are used.
    cudas: int | Iterable[int]
        IDs for CUDA device workers as an iterable, or an integer number of GPU workers to
        create. LiberTEM will use the IDs specified or assign round-robin to the available devices.
        In the iterable case these have to match CUDA device IDs on the system.
        Specify the same ID multiple times to spawn multiple workers on the same CUDA device.
    has_cupy: bool
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
        See https://docs.dask.org/en/stable/customize-initialization.html#preload-scripts
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

    if isinstance(cpus, int):
        cpus = tuple(range(cpus))

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

    cudas = assign_cudas(cudas)

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


def _run_task(task, params, task_id, threaded_executor, comms_topic: Optional[str]):
    """
    Very simple wrapper function. As dask internally caches functions that are
    submitted to the cluster in various ways, we need to make sure to
    consistently use the same function, and not build one on the fly.

    Without this function, UDFTask->UDF->UDFData ends up in the
    cache, which blows up memory usage over time.
    """
    worker_context = DaskWorkerContext(comms_topic)
    env = Environment(threads_per_worker=1,
                      threaded_executor=threaded_executor,
                      worker_context=worker_context)
    task_result = task(env=env, params=params)
    return {
        "task_result": task_result,
        "task_id": task_id,
    }


def _simple_run_task(task):
    return task()


class CommonDaskMixin:
    client: dd.Client

    def _task_idx_to_workers(self, workers: WorkerSet, idx: int) -> WorkerSet:
        hosts = list(sorted(workers.hosts()))
        host_idx = idx % len(hosts)
        host = hosts[host_idx]
        return workers.filter(lambda w: w.host == host)

    def _future_for_location(
        self,
        task: TaskProtocol,
        locations: WorkerSet,
        resources,
        workers: WorkerSet,
        task_args=None,
        wrap_fn=_run_task,
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
            location_names = locations.names()
        else:
            location_names = None
        submit_kwargs.update({
            'resources': self._validate_resources(workers, resources),
            'workers': location_names,
            'pure': False,
        })
        return self.client.submit(
            wrap_fn, task, *task_args, **submit_kwargs
        )

    def _validate_resources(self, workers, resources):
        # This is set in the constructor of DaskJobExecutor
        if self.lt_resources:
            if not self._resources_available(workers, resources):
                raise ResourceError("Requested resources not available in cluster:", resources)
            result = resources
        else:
            if 'CUDA' in resources:
                raise ResourceError(
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

    def _get_future(
        self,
        task: TaskProtocol,
        workers: WorkerSet,
        idx: int,
        params_handle,
        threaded_executor,
        comms_topic: Optional[str]
    ):
        if len(workers) == 0:
            raise RuntimeError("no workers available!")
        params_fut = self._scatter_map[params_handle]
        return self._future_for_location(
            task=task,
            locations=task.get_locations() or self._task_idx_to_workers(
                workers, idx
            ),
            resources=task.get_resources(),
            workers=workers,
            task_args=(
                params_fut,
                idx,
                threaded_executor,
                comms_topic,
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


def _dispatch_messages(subscribers: dict[str, list[Callable]], dask_message: tuple[float, dict]):
    """
    Unpacks the Dask message format and forwards the message
    to all subscribed callbacks for that topic (if any)
    """
    timestamp, message = dask_message
    true_topic = message.pop('topic')
    for handler in subscribers.get(true_topic, []):
        handler(true_topic, message)


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
        self._scatter_map = {}
        self._snooze_manager = None
        self._worker_spec = None
        self._subscriptions = SubscriptionManager()

    def _scale_down(self):
        """
        If possible, scale the cluster down to one worker
        using Dask's :code:`cluster.scale`. Normally called by
        :code:`SnoozeManager`. Returns immediately, though
        Dask may take some time to shut down the extra workers
        in the background. There is no way to ensure that the
        remaining worker is the service (or even a CPU) worker.

        :meta private:
        """
        if not self.is_local or self._worker_spec is None or self.client.cluster is None:
            return
        self.client.cluster.scale(n=1)
        # cluster.scale returns immediately while workers are shut down
        # in the background by Dask. To avoid a bad state where we
        # `_scale_up` too quickly after calling this function then shutdown
        # the cluster, wait at most a few seconds until the cluster has time
        # to enact the call to `scale` before we hassle it with more requests
        # See https://github.com/dask/distributed/pull/9064 for more info
        t0 = time.monotonic()
        t1 = t0 + 3  # seconds of settling time
        while len(self.client.cluster.workers) > 1 and time.monotonic() < t1:
            time.sleep(0.1)

    def _scale_up(self):
        """
        If possible, scale the cluster back up to its initial
        state using Dask's :code:`cluster.scale`. Normally called by
        :code:`SnoozeManager`. The initial worker spec is stored in
        :code:`self._worker_spec`, set by :code:`self._enable_snooze`,
        as Dask deletes its copy of worker spec when scaling down.

        This method blocks until the workers are up, though there is
        an internal Dask async function (:code:`_wait_for_workers`) which
        could be used for the web client.

        :meta private:
        """
        if not self.is_local or self._worker_spec is None or self.client.cluster is None:
            return
        self.client.cluster.worker_spec = copy.copy(self._worker_spec)
        self.client.cluster.scale(n=len(self._worker_spec))
        # Block until our scale request is complete
        # There is also an async client._wait_for_workers(n, timeout)
        # which could be used in the web client, but it is nominally internal
        self.client.wait_for_workers(len(self._worker_spec))

    def _enable_snooze(self, timeout: float, spec: dict):
        """
        Enable the automatic snoozing on this executor,
        with a snooze timeout in seconds.

        :code:`spec` is the worker spec used to create the cluster
        behind the executor, so that it can be re-supplied to Dask
        during cluster scale_up.

        :meta private:
        """
        if self._snooze_manager is not None:
            return
        self._snooze_manager = SnoozeManager(
            up=self._scale_up,
            down=self._scale_down,
            timeout=timeout,
            subscriptions=self._subscriptions,
        )
        self._worker_spec = copy.copy(spec)

    @property
    def snooze_manager(self):
        return self._snooze_manager

    def subscribe(self, topic: str, callback: Callable[[str, dict], None]) -> str:
        return self._subscriptions.subscribe(topic, callback)

    def unsubscribe(self, key: str) -> bool:
        return self._subscriptions.unsubscribe(key)

    @keep_alive_context
    @contextlib.contextmanager
    def scatter(self, obj):
        # an additional layer of indirection, because we want to be able to
        # redirect keys to new values
        handle = str(uuid.uuid4())
        try:
            fut = self.client.scatter(obj, broadcast=True, hash=False)
            self._scatter_map[handle] = fut
            yield handle
        finally:
            if handle in self._scatter_map:
                del self._scatter_map[handle]

    @keep_alive
    def scatter_update(self, handle, obj):
        fut = self.client.scatter(obj, broadcast=True, hash=False)
        self._scatter_map[handle] = fut

    @keep_alive
    def scatter_update_patch(self, handle, patch):
        fut = self._scatter_map[handle]

        def _do_patch(obj):
            if not hasattr(obj, 'patch'):
                raise TypeError(f'object is not patcheable: {obj}')
            obj.patch(patch)

        # can't `client.run` here, as that doesn't resolve the scatter future
        futures = []
        for worker in self.get_available_workers().names():
            futures.append(self.client.submit(_do_patch, fut, pure=False, workers=[worker]))
        for res in dd.as_completed(futures, loop=self.client.loop):
            pass

    @keep_alive
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

        try:
            topic_id = f'topic-{cancel_id}'
            # Wrap all subscriptions into single unique topic
            self.client.subscribe_topic(
                topic_id,
                functools.partial(_dispatch_messages,
                                  task_comm_handler.subscriptions)
            )
        except AttributeError:
            # Dask version does not support structured logs
            # Fall back to partition-level progress updates only
            topic_id = None

        for w in range(int(len(workers))):
            if not tasks_w_index:
                break
            idx, wrapped_task = tasks_w_index.pop(0)
            future = self._get_future(wrapped_task, workers, idx, params_handle,
                                      threaded_executor, topic_id)
            initial.append(future)
            self._futures[cancel_id].append(future)

        try:
            as_completed = dd.as_completed(initial, with_results=True, loop=self.client.loop)
            for future, result_wrap in as_completed:
                if future.cancelled():
                    log.debug(
                        "future %r is cancelled, stopping",
                        future,
                    )
                    del self._futures[cancel_id]
                    raise JobCancelledError()
                result = result_wrap['task_result']
                task = _id_to_task(result_wrap['task_id'])
                if tasks_w_index:
                    idx, wrapped_task = tasks_w_index.pop(0)
                    future = self._get_future(
                        wrapped_task, workers, idx, params_handle, threaded_executor, topic_id
                    )
                    as_completed.add(future)
                    self._futures[cancel_id].append(future)
                yield result, task
        finally:
            if cancel_id in self._futures:
                del self._futures[cancel_id]
            if topic_id is not None:
                self.client.unsubscribe_topic(topic_id)

    def cancel(self, cancel_id):
        log.debug("cancelling with cancel_id=`%s`", cancel_id)
        if cancel_id in self._futures:
            futures = self._futures[cancel_id]
            self.client.cancel(futures)

    @keep_alive
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

    @keep_alive
    def run_function(self, fn, *args, **kwargs):
        """
        run a callable :code:`fn` on any worker
        """
        fn_with_args = functools.partial(fn, *args, **kwargs)
        future = self.client.submit(fn_with_args, priority=1, pure=False)
        return future.result()

    @keep_alive
    def map(self, fn, iterable):
        """
        Run a callable :code:`fn` for each element in :code:`iterable`, on arbitrary worker nodes.

        Parameters
        ----------

        fn : callable
            Function to call. Should accept exactly one parameter.

        iterable : Iterable
            Which elements to call the function on.
        """
        return [future.result()
                for future in self.client.map(fn, iterable, pure=False)]

    @keep_alive
    def run_each_host(self, fn, *args, **kwargs):
        """
        Run a callable :code:`fn` once on each host, gathering all results into
        a dict host -> result
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

    @keep_alive
    def run_each_worker(self, fn, *args, **kwargs):
        # Client.run() creates issues on Windows and OS X with Python 3.6
        # FIXME workaround may not be needed anymore for Python 3.7+
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
            # Client.close won't close the Cluster itself because
            # we provided an external dd.SpecCluster
            self.client.close()
            # Manually close the cluster if not yet torn down
            # use getattr just in case cluster is already gone
            if getattr(self.client, 'cluster', None) is not None:
                self.client.cluster.close(timeout=30)
            if self.snooze_manager is not None:
                self.snooze_manager.close()
        # NOTE: distributed already registers atexit handlers for
        # both clients and clusters, this is here to allow manual closure
        # followed by creation of a new Executor without accumulating clusters

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
            client_kwargs: Optional[dict] = None, preload: Optional[tuple[str]] = None,
            snooze_timeout: Optional[float] = None):
        """
        Spin up a local dask cluster

        Parameters
        ----------
        spec
            Dask cluster spec, see
            https://distributed.dask.org/en/stable/api.html#distributed.SpecCluster
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

        dist_log_level = dask.config.get('distributed.logging.distributed', default=None)
        if dist_log_level is None:
            dist_log_level = cluster_kwargs['silence_logs']

        with set_num_threads_env(n=1), set_worker_log_level(dist_log_level):
            # Mitigation for https://github.com/dask/distributed/issues/6776
            with dask.config.set({"distributed.worker.profile.enabled": False}):
                cluster = dd.SpecCluster(workers=spec, **(cluster_kwargs or {}))
                client = dd.Client(cluster, **(client_kwargs or {}))
                client.wait_for_workers(len(spec))

        is_local = not client_kwargs['set_as_default']

        executor = cls(client=client, is_local=is_local, lt_resources=True)
        if snooze_timeout is not None:
            executor._enable_snooze(snooze_timeout, spec)
        return executor

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
    scheduler,
    local_directory,
    cpus,
    cudas,
    has_cupy,
    name,
    log_level,
    preload: tuple[str, ...]
):
    import asyncio

    options = {
        "silence_logs": log_level,
        "local_directory": local_directory
    }

    spec = cluster_spec(
        cpus=cpus, cudas=cudas, has_cupy=has_cupy, name=name,
        options=options, preload=preload,
    )

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
