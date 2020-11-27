from functools import partial
import sys
import asyncio

import tornado
import tornado.web
import tornado.gen
import tornado.websocket
import tornado.ioloop
import tornado.escape
import tornado.ioloop


async def run_blocking(fn, *args, **kwargs):
    """
    run blocking function fn with args, kwargs in a thread and return a corresponding future
    """
    return await tornado.ioloop.IOLoop.current().run_in_executor(None, partial(fn, *args, **kwargs))


def adjust_event_loop_policy():
    # stolen from ipykernel:
    if sys.platform.startswith("win") and sys.version_info >= (3, 8):
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
