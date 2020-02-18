from functools import partial

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
