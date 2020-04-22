import os
import logging
import asyncio
import signal
import webbrowser
from functools import partial

import tornado.web
import tornado.gen
import tornado.websocket
import tornado.ioloop
import tornado.escape

from .state import SharedState
from .config import ConfigHandler
from .dataset import DataSetDetailHandler, DataSetDetectHandler, DataSetOpenSchema
from .browse import LocalFSBrowseHandler
from .jobs import JobDetailHandler
from .events import ResultEventHandler, EventRegistry
from .connect import ConnectHandler
from .analysis import AnalysisDetailHandler, DownloadDetailHandler, CompoundAnalysisHandler


log = logging.getLogger(__name__)


class IndexHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    def get(self):
        self.render("client/index.html")


def make_app(event_registry, shared_state):
    settings = {
        "static_path": os.path.join(os.path.dirname(__file__), "client"),
    }
    return tornado.web.Application([
        (r"/", IndexHandler, {"state": shared_state, "event_registry": event_registry}),
        (r"/api/datasets/detect/", DataSetDetectHandler, {
            "state": shared_state,
            "event_registry": event_registry
        }),
        (r"/api/datasets/schema/", DataSetOpenSchema, {
            "state": shared_state,
            "event_registry": event_registry
        }),
        (r"/api/datasets/([^/]+)/", DataSetDetailHandler, {
            "state": shared_state,
            "event_registry": event_registry
        }),
        (r"/api/browse/localfs/", LocalFSBrowseHandler, {
            "state": shared_state,
            "event_registry": event_registry
        }),
        (r"/api/jobs/([^/]+)/", JobDetailHandler, {
            "state": shared_state,
            "event_registry": event_registry
        }),
        (r"/api/compoundAnalyses/([^/]+)/analyses/([^/]+)/", AnalysisDetailHandler, {
            "state": shared_state,
            "event_registry": event_registry
        }),
        (r"/api/compoundAnalyses/([^/]+)/analyses/([^/]+)/download/([^/]+)/",
        DownloadDetailHandler, {
            "state": shared_state,
            "event_registry": event_registry
        }),
        (r"/api/compoundAnalyses/([^/]+)/", CompoundAnalysisHandler, {
            "state": shared_state,
            "event_registry": event_registry
        }),
        (r"/api/events/", ResultEventHandler, {
            "state": shared_state,
            "event_registry": event_registry
        }),
        (r"/api/config/", ConfigHandler, {
            "state": shared_state,
            "event_registry": event_registry
        }),
        (r"/api/config/connection/", ConnectHandler, {
            "state": shared_state,
            "event_registry": event_registry,
        }),
    ], **settings)


async def do_stop(shared_state):
    log.warning("Exiting...")
    log.debug("closing executor")
    if shared_state.executor_state.executor is not None:
        await shared_state.executor_state.executor.close()
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


def sig_exit(signum, frame, shared_state):
    log.info("Handling sig_exit...")
    loop = tornado.ioloop.IOLoop.instance()

    loop.add_callback_from_signal(
        lambda: asyncio.ensure_future(do_stop(shared_state))
    )


def main(host, port, event_registry, shared_state):
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    )
    log.info("listening on %s:%s" % (host, port))
    app = make_app(event_registry, shared_state)
    app.listen(address=host, port=port)
    return app


def run(host, port, browser, local_directory):
    # shared state:
    event_registry = EventRegistry()
    shared_state = SharedState()

    shared_state.set_local_directory(local_directory)
    main(host, port, event_registry, shared_state)
    if browser:
        webbrowser.open(f'http://{host}:{port}')
    loop = asyncio.get_event_loop()
    signal.signal(signal.SIGINT, partial(sig_exit, shared_state=shared_state))
    # Strictly necessary only on Windows, but doesn't do harm in any case.
    # FIXME check later if the unknown root cause was fixed upstream
    asyncio.ensure_future(nannynanny())
    loop.run_forever()


if __name__ == "__main__":
    main("0.0.0.0", 9000)
