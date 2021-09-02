import functools
import asyncio


async def sync_to_async(fn, pool=None, *args, **kwargs):
    """
    Run blocking function with `*args`, `**kwargs` in a thread pool.

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
    fn = functools.partial(fn, *args, **kwargs)
    return await loop.run_in_executor(pool, fn)
