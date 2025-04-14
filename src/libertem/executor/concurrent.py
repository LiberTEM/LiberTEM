import contextlib
import functools
import logging
import uuid
from typing import Optional
import concurrent.futures
from typing import Any
from collections.abc import Iterable

from opentelemetry import trace

from .base import (
    BaseJobExecutor, AsyncAdapter,
)
from libertem.common.executor import (
    JobCancelledError, TaskProtocol, TaskCommHandler, WorkerContext,
    SimpleWorkerQueue, WorkerQueue, Environment
)
from libertem.common.async_utils import sync_to_async
from libertem.utils.devices import detect
from libertem.common.scheduler import Worker, WorkerSet
from libertem.common.backend import get_use_cuda
from libertem.common.tracing import TracedThreadPoolExecutor


log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class ConcurrentWorkerContext(WorkerContext):
    def __init__(self, msg_queue: WorkerQueue):
        self._msg_queue = msg_queue

    def signal(self, ident: str, topic: str, msg_dict: dict[str, Any]):
        if 'ident' in msg_dict:
            raise ValueError('ident is a reserved name')
        msg_dict.update({'ident': ident})
        self._msg_queue.put((topic, msg_dict))


def _run_task(task, params, task_id, threaded_executor, msg_queue, scatter_map):
    """
    Wraps the task to be run in the pool
    """
    params = scatter_map[params]
    worker_context = ConcurrentWorkerContext(msg_queue)
    env = Environment(threads_per_worker=1,
                      threaded_executor=threaded_executor,
                      worker_context=worker_context)
    task_result = task(env=env, params=params)
    return {
        "task_result": task_result,
        "task_id": task_id,
    }


class ConcurrentJobExecutor(BaseJobExecutor):
    '''
    :class:`JobExecutor` that uses :mod:`python.concurrent.futures`.

    .. versionadded:: 0.9.0

    Parameters
    ----------

    client : concurrent.futures.Executor
    is_local : bool
        Shut the client down when the executor closes.
    '''
    def __init__(self, client: concurrent.futures.Executor, is_local=False):
        # Only import if actually instantiated, i.e. will likely be used
        import libertem.preload  # noqa: 401
        self.is_local = is_local
        self.client = client
        self._futures = {}
        self._scatter_map = {}

    @contextlib.contextmanager
    def scatter(self, obj):
        handle = str(uuid.uuid4())
        self._scatter_map[handle] = obj
        try:
            yield handle
        finally:
            del self._scatter_map[handle]

    def scatter_update(self, handle, obj):
        self._scatter_map[handle] = obj

    def scatter_update_patch(self, handle, patch):
        self._scatter_map[handle].patch(patch)

    def _get_future(self, wrapped_task, idx, params_handle, msg_queue):
        return self.client.submit(
            _run_task,
            task=wrapped_task, params=params_handle, task_id=idx,
            threaded_executor=True, msg_queue=msg_queue,
            scatter_map=self._scatter_map,
        )

    def run_tasks(
        self,
        tasks: Iterable[TaskProtocol],
        params_handle: Any,
        cancel_id: Any,
        task_comm_handler: TaskCommHandler,
    ):
        tasks = list(tasks)

        def _id_to_task(task_id):
            return tasks[task_id]

        self._futures[cancel_id] = []

        msg_queue = SimpleWorkerQueue()

        for idx, wrapped_task in list(enumerate(tasks)):
            future = self._get_future(wrapped_task, idx, params_handle, msg_queue)
            self._futures[cancel_id].append(future)

        with task_comm_handler.monitor(msg_queue):
            as_completed = None
            try:
                as_completed = concurrent.futures.as_completed(self._futures[cancel_id])
                for future in as_completed:
                    result_wrap = future.result()
                    if future.cancelled():
                        del self._futures[cancel_id]
                        raise JobCancelledError()
                    result = result_wrap['task_result']
                    task = _id_to_task(result_wrap['task_id'])
                    yield result, task
            finally:
                if cancel_id in self._futures:
                    del self._futures[cancel_id]
                if as_completed is not None:
                    as_completed.close()

    def cancel(self, cancel_id):
        if cancel_id in self._futures:
            for future in self._futures[cancel_id]:
                future.cancel()

    def run_function(self, fn, *args, **kwargs):
        """
        run a callable :code:`fn` on any worker
        """
        fn_with_args = functools.partial(fn, *args, **kwargs)
        future = self.client.submit(fn_with_args)
        return future.result()

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
        return self.client.map(fn, iterable)

    def get_available_workers(self):
        resources = {"compute": 1, "CPU": 1}
        if get_use_cuda() is not None:
            resources["CUDA"] = 1
            return WorkerSet([
                Worker(name='concurrent', host='localhost', resources=resources, nthreads=1)
            ])
        else:
            devices = detect()
            return WorkerSet([
                Worker(
                    name='concurrent', host='localhost',
                    resources=resources,
                    nthreads=len(devices['cpus']),
                )
            ])

    def run_each_host(self, fn, *args, **kwargs):
        """
        Run a callable :code:`fn` once on each host, gathering all results into
        a dict host -> result
        """
        # TODO: any cancellation/errors to handle?
        future = self.client.submit(fn, *args, **kwargs)
        return {"localhost": future.result()}

    def run_each_worker(self, fn, *args, **kwargs):
        future = self.client.submit(fn, *args, **kwargs)
        return {"inline": future.result()}

    def close(self):
        if self.is_local:
            self.client.shutdown(wait=False)

    @classmethod
    def make_local(cls, n_threads: Optional[int] = None):
        """
        Create a local ConcurrentJobExecutor backed by
        a :class:`python:concurrent.futures.ThreadPoolExecutor`

        Parameters
        ----------

        n_threads : Optional[int]
            The number of threads to spawn in the executor,
            by default None in which case as many threads as there
            are CPU cores will be spawned.

        Returns
        -------
        ConcurrentJobExecutor
            the connected JobExecutor
        """
        if n_threads is None:
            devices = detect()
            n_threads = len(devices['cpus'])
        client = TracedThreadPoolExecutor(tracer, max_workers=n_threads)
        return cls(client=client, is_local=True)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AsyncConcurrentJobExecutor(AsyncAdapter):
    def __init__(self, wrapped=None, *args, **kwargs):
        if wrapped is None:
            wrapped = ConcurrentJobExecutor(*args, **kwargs)
        super().__init__(wrapped)

    @classmethod
    async def make_local(cls):
        executor = await sync_to_async(functools.partial(
            ConcurrentJobExecutor.make_local,
        ))
        return cls(wrapped=executor)
