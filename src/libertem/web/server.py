import os
import logging
import asyncio
import signal
from functools import partial

import tornado.web
import tornado.gen
import tornado.websocket
import tornado.ioloop
import tornado.escape

from .base import SharedData
from .config import ConfigHandler
from .dataset import DataSetDetailHandler, DataSetDetectHandler
from .browse import LocalFSBrowseHandler
from .jobs import JobDetailHandler
from .events import ResultEventHandler, EventRegistry
from .connect import ConnectHandler


log = logging.getLogger(__name__)


class IndexHandler(tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    def get(self):
        self.render("client/index.html")


def make_app(event_registry, shared_data):
    settings = {
        "static_path": os.path.join(os.path.dirname(__file__), "client"),
    }
    return tornado.web.Application([
        (r"/", IndexHandler, {"data": shared_data, "event_registry": event_registry}),
        (r"/api/datasets/detect/", DataSetDetectHandler, {
            "data": shared_data,
            "event_registry": event_registry
        }),
        (r"/api/datasets/([^/]+)/", DataSetDetailHandler, {
            "data": shared_data,
            "event_registry": event_registry
        }),
        (r"/api/browse/localfs/", LocalFSBrowseHandler, {
            "data": shared_data,
            "event_registry": event_registry
        }),
        (r"/api/jobs/([^/]+)/", JobDetailHandler, {
            "data": shared_data,
            "event_registry": event_registry
        }),
        (r"/api/events/", ResultEventHandler, {
            "data": shared_data,
            "event_registry": event_registry
        }),
        (r"/api/config/", ConfigHandler, {
            "data": shared_data,
            "event_registry": event_registry
        }),
        (r"/api/config/connection/", ConnectHandler, {
            "data": shared_data,
            "event_registry": event_registry,
        }),
    ], **settings)


async def do_stop(shared_data):
    log.warning("Exiting...")
    log.debug("closing executor")
    if shared_data.executor is not None:
        await shared_data.executor.close()
    loop = asyncio.get_event_loop()
    log.debug("stopping event loop")
    loop.stop()


async def nannynanny():
    '''
    Make sure the event loop wakes up regularly.

    This mitigates a strange bug on Windows
    where Ctrl-C is only handled after an event is processed.

    See Issue #356
    '''
    while True:
        await asyncio.sleep(1)


def sig_exit(signum, frame, shared_data):
    log.info("Handling sig_exit...")
    loop = tornado.ioloop.IOLoop.instance()

    loop.add_callback_from_signal(
        lambda: asyncio.ensure_future(do_stop(shared_data))
    )


def main(host, port, event_registry, shared_data):
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    )
    log.info("listening on %s:%s" % (host, port))
    app = make_app(event_registry, shared_data)
    app.listen(address=host, port=port)
    return app


def run(host, port, local_directory):
    # shared state:
    event_registry = EventRegistry()
    shared_data = SharedData()

    shared_data.set_local_directory(local_directory)
    main(host, port, event_registry, shared_data)
    loop = asyncio.get_event_loop()
    signal.signal(signal.SIGINT, partial(sig_exit, shared_data=shared_data))
    # Strictly necessary only on Windows, but doesn't do harm in any case.
    # FIXME check later if the unknown root cause was fixed upstream
    asyncio.ensure_future(nannynanny())
    loop.run_forever()


if __name__ == "__main__":
    main("0.0.0.0", 9000)
