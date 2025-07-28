import functools
import asyncio
from typing import Callable, TYPE_CHECKING, TypeVar, Optional

from contextlib import asynccontextmanager

from libertem.common.async_utils import (
    adjust_event_loop_policy, sync_to_async, async_generator_eager
)
from libertem.common.executor import (
    JobExecutor, AsyncJobExecutor,
    Environment, GenericTaskProtocol, GPUSpec,
)
from libertem.common.tracing import TracedThreadPoolExecutor
from libertem.common.exceptions import ExecutorSpecException


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


def make_canonical(main_gpu: GPUSpec) -> Optional[int]:
    '''
    Handle default cases when specifying a main GPU

    .. versionadded:: 0.16.0

    Parameters
    ----------

    main_gpu : int or bool, optional

        GPU spec to use for GPU processing on the main process.

        True:
            Use any available GPU, throw an error if none are available.
        int:
            Specify GPU ID to use, throw an error if it is not present.
        False:
            Don't use GPU on the main process.
        None:
            Default behavior. Currently activates GPU processing if a GPU is
            available to catch any potential issues with this feature.

    Returns
    -------

    int or None
        GPU ID or :code:`None` for no GPU processing
    '''
    if main_gpu is False:
        main_gpu = None
    elif main_gpu is True:
        from libertem.utils.devices import detect
        detect_out = detect()
        if detect_out['cudas'] and detect_out['has_cupy']:
            main_gpu = detect_out['cudas'][0]
        else:
            raise ExecutorSpecException(
                'Cannot specify main GPU as no GPUs detected or no CuPy found.'
            )
    elif main_gpu is None:
        # Activate if available
        from libertem.utils.devices import detect
        detect_out = detect()
        if detect_out['cudas'] and detect_out['has_cupy']:
            main_gpu = detect_out['cudas'][0]
    # check last because instanceof(<bool>, int) is True
    elif isinstance(main_gpu, int):
        # Verify it is a valid GPU ID
        from libertem.utils.devices import detect
        detect_out = detect()
        if main_gpu not in detect_out['cudas'] or not detect_out['has_cupy']:
            raise ExecutorSpecException(
                f"Main GPU {main_gpu} not detected, "
                f"only found {detect_out['cudas']}."
            )
    else:
        raise ValueError(f'Invalid type for main_process_gpu: {type(main_gpu)}')
    return main_gpu


class BaseJobExecutor(JobExecutor):
    '''
    Base class for LiberTEM executors

    Contains a generic implementation for
    :meth:`libertem.common.executor.JobExecutor.run_process_local` for re-use in
    executors that don't implement a specialized version.

    Parameters
    ----------

    main_process_gpu : int or None, optional
        GPU to set in the :class:`~libertem.common.executor.Environment`
        supplied to the task in :meth:`run_process_local`.
    '''
    def __init__(self, main_process_gpu: Optional[int] = None):
        self._main_process_gpu = main_process_gpu

    def get_udf_runner(self) -> type['UDFRunner']:
        from libertem.udf.base import UDFRunner
        return UDFRunner

    def ensure_async(self, pool=None):
        """
        Returns an asynchronous executor; by default just wrap into `AsyncAdapter`.
        """
        return AsyncAdapter(wrapped=self, pool=pool)

    def run_process_local(self, task: GenericTaskProtocol, args=(), kwargs: Optional[dict] = None):
        """
        run a callable :code:`fn` in the context of the current process.
        """
        if kwargs is None:
            kwargs = {}
        env = self._get_local_env()
        return task(args, kwargs, environment=env)

    def _get_local_env(self):
        return Environment(
            threads_per_worker=None,
            threaded_executor=True,
            gpu_id=self._main_process_gpu
        )


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
