import concurrent
import functools
import asyncio


class ExecutorError(Exception):
    pass


class JobCancelledError(Exception):
    """
    raised by async executors in run_job if the job was cancelled
    """
    pass


class JobExecutor(object):
    def run_job(self, job, cancel_id=None):
        """
        run a Job
        """
        raise NotImplementedError()

    def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn`
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


class AsyncJobExecutor(object):
    async def run_job(self, job, cancel_id):
        """
        run a Job
        """
        raise NotImplementedError()

    def run_tasks(self, tasks, cancel_id):
        """
        run a number of Tasks
        """
        raise NotImplementedError()

    async def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn`
        """
        raise NotImplementedError()

    async def close(self):
        """
        cleanup resources used by this executor, if any
        """

    async def cancel(self, cancel_id):
        """
        cancel execution identified by `cancel_id`
        """
        pass

    async def get_available_workers(self):
        raise NotImplementedError()


async def sync_to_async(fn, pool=None):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(pool, fn)


async def async_generator(gen, pool=None):
    def inner_next(gen):
        try:
            return next(gen)
        except StopIteration:
            raise MyStopIteration()

    loop = asyncio.get_event_loop()
    while True:
        try:
            item = await loop.run_in_executor(pool, inner_next, gen)
            yield item
        except MyStopIteration:
            break


class MyStopIteration(Exception):
    """
    TypeError: StopIteration interacts badly with generators
    and cannot be raised into a Future
    """
    pass


class AsyncAdapter(AsyncJobExecutor):
    def __init__(self, wrapped):
        """
        Wrap a synchronous JobExecutor and allow to use it as AsyncJobExecutor. All methods are
        converted to async and executed in a separate thread.
        """
        self._wrapped = wrapped
        self._pool = concurrent.futures.ThreadPoolExecutor(1)

    async def run_job(self, job, cancel_id):
        """
        run a Job
        """
        gen = self._wrapped.run_job(job, cancel_id)
        agen = async_generator(gen, self._pool)
        async for i in agen:
            yield i

    async def run_tasks(self, tasks, cancel_id):
        """
        run a number of Tasks
        """
        gen = self._wrapped.run_tasks(tasks, cancel_id)
        agen = async_generator(gen, self._pool)
        async for i in agen:
            yield i

    async def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn`
        """
        fn_with_args = functools.partial(self._wrapped.run_function, fn, *args, **kwargs)
        return await sync_to_async(fn_with_args, self._pool)

    async def close(self):
        """
        cleanup resources used by this executor, if any
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
