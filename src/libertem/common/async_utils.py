import sys
import queue
import asyncio
import threading
from typing import Callable, Optional, TypeVar
from collections.abc import AsyncGenerator, Generator
from concurrent.futures import ThreadPoolExecutor

from opentelemetry import trace

from libertem.common.tracing import attach_to_parent

tracer = trace.get_tracer(__name__)
T = TypeVar('T')


class MyStopIteration(Exception):
    """
    TypeError: StopIteration interacts badly with generators
    and cannot be raised into a Future
    """
    pass


def _wrap_f(f, args, kwargs):
    def _wrapped():
        try:
            return f(*args, **kwargs)
        except StopIteration as e:
            raise MyStopIteration() from e
    return _wrapped


async def sync_to_async(
    fn: Callable[..., T], pool: Optional[ThreadPoolExecutor] = None, *args, **kwargs
) -> T:
    """
    Run blocking function with `*args`, `**kwargs` in a thread pool.

    Raises `MyStopIteration` in case `fn` raises a `StopIteration`.

    Parameters
    ----------
    fn : callable
        The blocking function to run in a background thread

    pool : ThreadPoolExecutor or None
        In which thread pool should the function be run? If `None`, we create a new one

    *args, **kwargs
        Passed on to `fn`
    """
    loop = asyncio.get_event_loop()
    # As `fn` may throw a `StopIteration` for whatever reason, which doesn't
    # work inside of futures, we have to wrap it in a custom exception type.
    # Without this, it's possible that things hang instead of propagating
    # errors properly:
    fn = _wrap_f(fn, args, kwargs)
    return await loop.run_in_executor(pool, fn)


async def async_generator(
    gen: Generator[T, None, None], pool: Optional[ThreadPoolExecutor] = None
) -> AsyncGenerator[T, None]:
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


async def run_agen_get_last(gen: AsyncGenerator[T, None]) -> T:
    result = None
    try:
        while True:
            result = await gen.__anext__()
    except StopAsyncIteration:
        pass
    finally:
        await gen.aclose()
    if result is None:
        raise RuntimeError("run_agen_get_last called with empty generator")
    return result


def run_gen_get_last(gen: Generator[T, None, None]) -> T:
    result = None
    try:
        while True:
            result = gen.__next__()
    except StopIteration:
        pass
    finally:
        gen.close()
    if result is None:
        raise RuntimeError("run_gen_get_last called with empty generator")
    return result


class AsyncGenToQueueThread(threading.Thread):
    """
    Wrap an async generator and execute it in a thread, putting the generated
    items into a queue.

    Parameters
    ----------
    agen : iterable
        The async generator to wrap

    q : queue.Queue
        The result queue where the generated items will be put. The calling
        thread should consume the items in this queue.

    args, kwargs
        will be passed to :code:`Thread.__init__`
    """
    def __init__(self, agen, q, span_context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._agen = agen
        self._q = q
        self._should_stop = threading.Event()
        self._span_context = span_context
        self.ex = None

    def run(self):
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            adjust_event_loop_policy()

            async def _wrap_gen(agen):
                async for item in agen:
                    self._q.put(item)
                    if self._should_stop.is_set():
                        break
            loop.run_until_complete(_wrap_gen(self._agen))
        except Exception as e:
            self.ex = e
        finally:
            self._q.put(MyStopIteration)
        return

    def get_exception(self):
        return self.ex

    def stop(self):
        self._should_stop.set()


def async_to_sync_generator(agen, pool=None):
    span = trace.get_current_span()
    span_context = span.get_span_context()
    q = queue.Queue()
    t = AsyncGenToQueueThread(agen, q, span_context)
    t.start()
    try:
        while True:
            item = q.get()
            if item is MyStopIteration:
                # propagate any uncaught exception in the wrapped generator to the calling thread:
                ex = t.get_exception()
                if ex is not None:
                    raise ex
                break
            yield item

        if q.qsize() > 0:
            res = q.get()
            if isinstance(res, Exception):
                raise res  # TODO: re-raise in a better way?
    finally:
        t.join()


class SyncGenToQueueThread(threading.Thread):
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
    def __init__(self, gen, q, span_context, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gen = gen
        self._q = q
        self._should_stop = threading.Event()
        self._span_context = span_context
        self.ex = None

    def run(self):
        with attach_to_parent(self._span_context):
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
    span = trace.get_current_span()
    span_context = span.get_span_context()
    q = queue.Queue()
    t = SyncGenToQueueThread(gen, q, span_context)

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
        # in case our thread raises an exception, we may need to stop the `SyncGenToQueueThread`:
        t.stop()
        t.join()


def adjust_event_loop_policy():
    """
    Set an appropriate event loop policy on Windows. The new one from Python 3.8 doesn't
    work for us by default, so call this as early as possible!
    """
    # stolen from ipykernel:
    if sys.platform.startswith("win"):
        try:
            from asyncio import (
                WindowsProactorEventLoopPolicy,
                WindowsSelectorEventLoopPolicy,
            )
        except ImportError:
            # not affected
            pass
        else:
            # FIXME this might fail if the event loop policy has been overridden by something
            # incompatible that is not a WindowsProactorEventLoopPolicy. Dask.distributed creates
            # a custom event loop policy, for example. Fortunately that one should be compatible.
            # Better would be "duck typing", i.e. to check if the required add_reader() method
            # of the loop throws a NotImplementedError.
            if type(asyncio.get_event_loop_policy()) is WindowsProactorEventLoopPolicy:
                # WindowsProactorEventLoopPolicy is not compatible with tornado 6
                # fallback to the pre-3.8 default of Selector
                asyncio.set_event_loop_policy(WindowsSelectorEventLoopPolicy())
