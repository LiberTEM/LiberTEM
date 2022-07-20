import contextlib
import functools
import logging
import concurrent.futures
from typing import Iterable, Any

from opentelemetry import trace

from .base import (
    BaseJobExecutor, AsyncAdapter,
)
from libertem.common.executor import (
    JobCancelledError, TaskProtocol, TaskCommHandler,
)
from libertem.common.async_utils import sync_to_async
from libertem.utils.devices import detect
from libertem.common.scheduler import Worker, WorkerSet
from libertem.common.backend import get_use_cuda
from libertem.common.tracing import TracedThreadPoolExecutor
from .dask import _run_task


log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


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

    @contextlib.contextmanager
    def scatter(self, obj):
        yield obj

    def _get_future(self, wrapped_task, idx, params_handle):
        return self.client.submit(
            _run_task,
            task=wrapped_task, params=params_handle, task_id=idx, threaded_executor=True
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

        for idx, wrapped_task in list(enumerate(tasks)):
            future = self._get_future(wrapped_task, idx, params_handle)
            self._futures[cancel_id].append(future)

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

    def cancel(self, cancel_id):
        if cancel_id in self._futures:
            for future in self._futures[cancel_id]:
                future.cancel()

    def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn` on any worker
        """
        fn_with_args = functools.partial(fn, *args, **kwargs)
        future = self.client.submit(fn_with_args)
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
        Run a callable `fn` once on each host, gathering all results into a dict host -> result
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
    def make_local(cls):
        """
        Create a local ConcurrentJobExecutor backed by
        a :class:`python:concurrent.futures.ThreadPoolExecutor`

        Returns
        -------
        ConcurrentJobExecutor
            the connected JobExecutor
        """
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
