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
    def run_job(self, job):
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
        returns a list of dicts with available workers
        keys of the dictionary:
            name : the identifying name of the worker
            host : ip address or hostname where the worker is running
        """
        raise NotImplementedError()


class AsyncJobExecutor(object):
    async def run_job(self, job):
        """
        run a Job
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

    async def cancel_job(self, job):
        """
        cancel execution of `job`
        """
        pass

    async def get_available_workers(self):
        raise NotImplementedError()


async def sync_to_async(fn):
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        return await loop.run_in_executor(pool, fn)


async def async_generator(gen):
    def inner_next(gen):
        try:
            return next(gen)
        except StopIteration:
            raise MyStopIteration()

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
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

    async def run_job(self, job):
        """
        run a Job
        """
        gen = self._wrapped.run_job(job)
        agen = async_generator(gen)
        async for i in agen:
            yield i

    async def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn`
        """
        fn_with_args = functools.partial(self._wrapped.run_function, fn, *args, **kwargs)
        return await sync_to_async(fn_with_args)

    async def close(self):
        """
        cleanup resources used by this executor, if any
        """
        return await sync_to_async(self._wrapped.close)

    async def cancel_job(self, job):
        """
        cancel execution of `job`
        """
        return await sync_to_async(functools.partial(self._wrapped.cancel_job, job=job))
