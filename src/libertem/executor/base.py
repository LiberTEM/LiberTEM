import concurrent
import functools
import asyncio
from typing import Optional, Any, Iterable
from typing_extensions import Protocol
from contextlib import contextmanager
from async_generator import asynccontextmanager

from libertem.utils.threading import set_num_threads
from libertem.utils.async_utils import (
    adjust_event_loop_policy, sync_to_async, async_generator_eager
)


class ExecutorError(Exception):
    pass


class JobCancelledError(Exception):
    """
    raised by async executors in run_tasks() or run_each_partition() if the task was cancelled
    """
    pass


class Environment:
    def __init__(self, threads_per_worker):
        self._threads_per_worker = threads_per_worker

    @property
    def threads_per_worker(self) -> Optional[int]:
        """
        int or None : number of threads that a UDF is allowed to use in the `process_*` method.
                      For numba, pyfftw, OMP, MKL, OpenBLAS, this limit is set automatically;
                      this property can be used for other cases, like manually creating
                      thread pools.
                      None means no limit is set, and the UDF can use any number of threads
                      it deems necessary (should be limited to system limits, of course).

        See also: :func:`libertem.utils.threading.set_num_threads`

        .. versionadded:: 0.7.0
        """
        return self._threads_per_worker

    @contextmanager
    def enter(self):
        """
        Note: we are using the @contextmanager decorator here,
        because with separate `__enter__`, `__exit__` methods,
        we can't easily delegate to `set_num_threads`, or other
        contextmanagers that may come later.
        """
        with set_num_threads(self._threads_per_worker):
            yield self


class TaskProtocol(Protocol):
    def __call__(self, env: Environment, params: Any):
        pass


class JobExecutor:
    def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn` on any worker
        """
        raise NotImplementedError()

    @contextmanager
    def scatter(self, obj):
        '''
        Scatter :code:`obj` throughout the cluster

        Parameters
        ----------

        obj
            Some kind of Python object or variable

        Returns
        -------
        handle
            Handle for the scattered :code:`obj`
        '''
        raise NotImplementedError()

    def run_tasks(
        self,
        tasks: Iterable[TaskProtocol],
        params_handle: Any,
        cancel_id: Any,
    ):
        """
        Run the tasks with the given parameters

        Parameters
        ----------
        tasks
            The tasks to be run
        params_handle : [type]
            A handle for the task parameters, as returned from :meth:`JobExecutor.scatter`
        cancel_id
            An identifier which can be used for cancelling all tasks together. The
            same identifier should be passed to :meth:`AsyncJobExecutor.cancel`
        """
        raise NotImplementedError()

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
        raise NotImplementedError()

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
        raise NotImplementedError()

    def run_each_host(self, fn, *args, **kwargs):
        """
        Run a callable `fn` once on each host, gathering all results into a dict host -> result


        Parameters
        ----------

        fn : callable
            Function to call

        *args
            Arguments for fn

        **kwargs
            Keyword arguments for fn
        """
        raise NotImplementedError()

    def run_each_worker(self, fn, *args, **kwargs):
        """
        Run `fn` on each worker process, and pass *args, **kwargs to it.

        Useful, for example, if you need to prepare the environment of
        each Python interpreter, warm up per-process caches etc.

        Parameters
        ----------
        fn : callable
            Function to call

        *args
            Arguments for fn

        **kwargs
            Keyword arguments for fn

        Returns
        -------
        dict
            Return values keyed by worker name (executor-specific)
        """
        raise NotImplementedError()

    def close(self):
        """
        cleanup resources used by this executor, if any
        """

    def get_available_workers(self):
        """
        Returns a list of dicts with available workers

        keys of the dictionary:
            name : an identifying name of the worker
            host : ip address or hostname where the worker is running

        Each worker should correspond to a "worker process", so if the executor
        is using multiple processes or threads, each process/thread should be
        included in this list.
        """
        raise NotImplementedError()

    def get_resource_details(self):
        """
        Returns a list of dicts with cluster details

        key of the dictionary:
            host: ip address or hostname where the worker is running
        """
        raise NotImplementedError()

    def ensure_sync(self):
        """
        Returns a synchronous executor, incase of a `JobExecutor` we just
        return `self; in case of `AsyncJobExecutor` below more work is needed!
        """
        return self

    def ensure_async(self, pool=None):
        """
        Returns an asynchronous executor; by default just wrap into `AsyncAdapter`.
        """
        return AsyncAdapter(wrapped=self, pool=pool)


class AsyncJobExecutor:
    async def run_tasks(self, tasks, params_handle, cancel_id):
        """
        Run a number of Tasks, yielding (result, task) tuples
        """
        raise NotImplementedError()

    async def run_function(self, fn, *args, **kwargs):
        """
        Run a callable `fn` on any worker
        """
        raise NotImplementedError()

    async def run_each_partition(self, partitions, fn, all_nodes=False):
        raise NotImplementedError()

    async def map(self, fn, iterable):
        """
        Run a callable `fn` for each item in iterable, on arbitrary worker nodes

        Parameters
        ----------

        fn : callable
            Function to call. Should accept exactly one parameter.

        iterable : Iterable
            Which elements to call the function on.
        """
        raise NotImplementedError()

    async def run_each_host(self, fn, *args, **kwargs):
        raise NotImplementedError()

    async def run_each_worker(self, fn, *args, **kwargs):
        """
        Run `fn` on each worker process, and pass *args, **kwargs to it.

        Useful, for example, if you need to prepare the environment of
        each Python interpreter, warm up per-process caches etc.

        Parameters
        ----------
        fn : callable
            Function to call

        *args
            Arguments for fn

        **kwargs
            Keyword arguments for fn
        """
        raise NotImplementedError()

    async def close(self):
        """
        Cleanup resources used by this executor, if any.
        """

    async def cancel(self, cancel_id):
        """
        cancel execution identified by `cancel_id`
        """
        pass

    async def get_available_workers(self):
        raise NotImplementedError()

    async def get_resource_details(self):
        raise NotImplementedError()

    def ensure_sync(self):
        raise NotImplementedError()

    def ensure_async(self, pool=None):
        """
        Returns an asynchronous executor; by default just return `self`.
        """
        return self


class AsyncAdapter(AsyncJobExecutor):
    """
    Wrap a synchronous JobExecutor and allow to use it as AsyncJobExecutor. All methods are
    converted to async and executed in a separate thread.
    """
    def __init__(self, wrapped, pool=None):
        self._wrapped = wrapped
        if pool is None:
            pool = AsyncAdapter.make_pool()
        self._pool = pool

    @classmethod
    def make_pool(cls):
        pool = concurrent.futures.ThreadPoolExecutor(1)
        pool.submit(adjust_event_loop_policy).result()
        pool.submit(lambda: asyncio.set_event_loop(asyncio.new_event_loop())).result()
        return pool

    def ensure_sync(self):
        return self._wrapped

    @asynccontextmanager
    async def scatter(self, obj):
        try:
            res = await sync_to_async(self._wrapped.scatter.__enter__, self._pool)
            yield res
        finally:
            exit_fn = functools.partial(
                self._wrapped.scatter.__exit__,
                None, None, None,  # FIXME: exc_type, exc_value, traceback?
            )
            await sync_to_async(exit_fn, self._pool)

    async def run_tasks(self, tasks, params_handle, cancel_id):
        """
        run a number of Tasks
        """
        gen = self._wrapped.run_tasks(tasks, params_handle, cancel_id)
        agen = async_generator_eager(gen, self._pool)
        async for i in agen:
            yield i

    async def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn` on an arbitrary worker node
        """
        fn_with_args = functools.partial(self._wrapped.run_function, fn, *args, **kwargs)
        return await sync_to_async(fn_with_args, self._pool)

    async def run_each_partition(self, partitions, fn, all_nodes=False):
        fn_with_args = functools.partial(
            self._wrapped.run_each_partition, partitions, fn, all_nodes
        )
        return await sync_to_async(fn_with_args, self._pool)

    async def map(self, fn, iterable):
        """
        Run a callable `fn` for each item in iterable, on arbitrary worker nodes

        Parameters
        ----------

        fn : callable
            Function to call. Should accept exactly one parameter.

        iterable : Iterable
            Which elements to call the function on.
        """
        fn_with_args = functools.partial(
            self._wrapped.map, fn, iterable,
        )
        return await sync_to_async(fn_with_args, self._pool)

    async def run_each_host(self, fn, *args, **kwargs):
        fn_with_args = functools.partial(self._wrapped.run_each_host, fn, *args, **kwargs)
        return await sync_to_async(fn_with_args, self._pool)

    async def run_each_worker(self, fn, *args, **kwargs):
        fn_with_args = functools.partial(self._wrapped.run_each_worker, fn, *args, **kwargs)
        return await sync_to_async(fn_with_args, self._pool)

    async def close(self):
        """
        Cleanup resources used by this executor, if any, including the wrapped executor.
        """
        res = await sync_to_async(self._wrapped.close, self._pool)
        if self._pool:
            self._pool.shutdown()
        return res

    async def cancel(self, cancel_id):
        """
        cancel execution identified by cancel_id
        """
        return await sync_to_async(
            functools.partial(self._wrapped.cancel, cancel_id=cancel_id),
            self._pool
        )

    async def get_available_workers(self):
        return await sync_to_async(self._wrapped.get_available_workers)

    async def get_resource_details(self):
        return await sync_to_async(self._wrapped.get_resource_details)
