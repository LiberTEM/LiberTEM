import concurrent
import functools
import asyncio
import threading
import queue

from libertem.utils.async_utils import adjust_event_loop_policy


class ExecutorError(Exception):
    pass


class JobCancelledError(Exception):
    """
    raised by async executors in run_tasks() or run_each_partition() if the task was cancelled
    """
    pass


class JobExecutor(object):
    def run_function(self, fn, *args, **kwargs):
        """
        run a callable `fn` on any worker
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


class AsyncJobExecutor(object):
    async def run_tasks(self, tasks, cancel_id):
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


class GenThread(threading.Thread):
    def __init__(self, gen, q, *args, **kwargs):
        """
        Wrap a generator and execute it in a thread, putting the generated
        items into a queue.

        Parameters
        ----------
        gen : iterable
            The generator to wrap

        q : queue.Queue
            The result queue where the generated items will be put. The calling
            thread should consume the items in this queue.

        args, kwargs
            will be passed to :code:`Thread.__init__`
        """
        super().__init__(*args, **kwargs)
        self._gen = gen
        self._q = q
        self._should_stop = threading.Event()
        self.ex = None

    def run(self):
        try:
            # looping over `self._gen` can take some time for each item,
            # this is where the "real" work happens:
            for item in self._gen:
                self._q.put(item)
                if self._should_stop.is_set():
                    break
        except Exception as e:
            self.ex = e
        finally:
            self._q.put(MyStopIteration)
        return

    def get_exception(self):
        return self.ex

    def stop(self):
        self._should_stop.set()


async def async_generator_eager(gen, pool=None):
    """
    Convert the synchronous generator `gen` to an async generator. Eagerly run
    `gen` in a thread and provide the result in a queue. This means that `gen`
    can run ahead of the asynchronous generator that is returned.

    Parameters:
    -----------

    gen : iterable
        The generator to run

    pool: ThreadPoolExecutor
        The thread pool to run the generator in, can be None to create an
        ad-hoc thread
    """
    q = queue.Queue()
    t = GenThread(gen, q)

    loop = asyncio.get_event_loop()

    t.start()
    try:
        while True:
            # get a single item from the result queue:
            item = await loop.run_in_executor(pool, q.get)

            # MyStopIteration is a canary value to signal that the inner
            # generator has finished running
            if item is MyStopIteration:
                # propagate any uncaught exception in the wrapped generator to the calling thread:
                ex = t.get_exception()
                if ex is not None:
                    raise ex
                break
            yield item
    finally:
        t.stop()  # in case our thread raises an exception, we may need to stop the `GenThread`:
        t.join()


class MyStopIteration(Exception):
    """
    TypeError: StopIteration interacts badly with generators
    and cannot be raised into a Future
    """
    pass


class AsyncAdapter(AsyncJobExecutor):
    def __init__(self, wrapped, pool=None):
        """
        Wrap a synchronous JobExecutor and allow to use it as AsyncJobExecutor. All methods are
        converted to async and executed in a separate thread.
        """
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

    async def run_tasks(self, tasks, cancel_id):
        """
        run a number of Tasks
        """
        gen = self._wrapped.run_tasks(tasks, cancel_id)
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
