import functools
import asyncio
from typing import Callable, TYPE_CHECKING, TypeVar, Optional

from contextlib import asynccontextmanager

from libertem.common.async_utils import (
    adjust_event_loop_policy, sync_to_async, async_generator_eager
)
from libertem.common.executor import JobExecutor, AsyncJobExecutor
from libertem.common.tracing import TracedThreadPoolExecutor


if TYPE_CHECKING:
    from libertem.udf.base import UDFRunner
    from libertem.common.snooze import SnoozeManager

T = TypeVar('T')


class ResourceError(RuntimeError):
    """
    Thrown when there is a resource mismatch, for example if the task requests
    resources that are not available in the worker pool.
    """
    pass


class BaseJobExecutor(JobExecutor):
    def get_udf_runner(self) -> type['UDFRunner']:
        from libertem.udf.base import UDFRunner
        return UDFRunner

    def ensure_async(self, pool=None):
        """
        Returns an asynchronous executor; by default just wrap into `AsyncAdapter`.
        """
        return AsyncAdapter(wrapped=self, pool=pool)


class AsyncAdapter(AsyncJobExecutor):
    """
    Wrap a synchronous JobExecutor and allow to use it as AsyncJobExecutor. All methods are
    converted to async and executed in a separate thread.
    """
    def __init__(self, wrapped: JobExecutor, pool=None):
        self._wrapped = wrapped
        if pool is None:
            pool = AsyncAdapter.make_pool()
        self._pool = pool

    @classmethod
    def make_pool(cls):
        pool = TracedThreadPoolExecutor(1)
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

    async def run_function(self, fn: Callable[..., T], *args, **kwargs) -> T:
        """
        run a callable :code:`fn` on an arbitrary worker node
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
        Run a callable :code:`fn` for each item in iterable, on arbitrary worker nodes

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

    def get_udf_runner(self) -> type['UDFRunner']:
        from libertem.udf.base import UDFRunner
        return UDFRunner

    @property
    def snooze_manager(self) -> Optional['SnoozeManager']:
        return self._wrapped.snooze_manager
