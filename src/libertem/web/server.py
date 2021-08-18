import os
import sys
import logging
import asyncio
import signal
import select
import threading
import webbrowser
from functools import partial

import tornado.web
import tornado.gen
import tornado.websocket
import tornado.ioloop
import tornado.escape

from .base import TokenAuthMixin
from .shutdown import ShutdownHandler
from .state import SharedState
from .config import ConfigHandler, ClusterDetailHandler
from .dataset import DataSetDetailHandler, DataSetDetectHandler, DataSetOpenSchema
from .browse import LocalFSBrowseHandler
from .jobs import JobDetailHandler
from .events import ResultEventHandler, EventRegistry
from .connect import ConnectHandler
from .analysis import AnalysisDetailHandler, DownloadDetailHandler, CompoundAnalysisHandler
from .generator import DownloadScriptHandler, CopyScriptHandler

log = logging.getLogger(__name__)


class IndexHandler(TokenAuthMixin, tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry, token):
        self.state = state
        self.event_registry = event_registry
        self.token = token

    def get(self):
        self.render("client/index.html")


def make_app(event_registry, shared_state, token=None):
    settings = {
        "static_path": os.path.join(os.path.dirname(__file__), "client"),
    }
    common_kwargs = {
        "state": shared_state,
        "event_registry": event_registry,
        "token": token,
    }
    return tornado.web.Application([
        (r"/", IndexHandler, common_kwargs),
        (r"/api/datasets/detect/", DataSetDetectHandler, common_kwargs),
        (r"/api/datasets/schema/", DataSetOpenSchema, common_kwargs),
        (r"/api/datasets/([^/]+)/", DataSetDetailHandler, common_kwargs),
        (r"/api/browse/localfs/", LocalFSBrowseHandler, common_kwargs),
        (r"/api/jobs/([^/]+)/", JobDetailHandler, common_kwargs),
        (r"/api/compoundAnalyses/([^/]+)/analyses/([^/]+)/", AnalysisDetailHandler, common_kwargs),
        (
            r"/api/compoundAnalyses/([^/]+)/analyses/([^/]+)/download/([^/]+)/",
            DownloadDetailHandler,
            common_kwargs
        ),
        (r"/api/compoundAnalyses/([^/]+)/copy/notebook/", CopyScriptHandler, common_kwargs),
        (r"/api/compoundAnalyses/([^/]+)/download/notebook/", DownloadScriptHandler, common_kwargs),
        (r"/api/compoundAnalyses/([^/]+)/", CompoundAnalysisHandler, common_kwargs),
        (r"/api/events/", ResultEventHandler, common_kwargs),
        (r"/api/shutdown/", ShutdownHandler, common_kwargs),
        (r"/api/config/", ConfigHandler, common_kwargs),
        (r"/api/config/cluster/", ClusterDetailHandler, common_kwargs),
        (r"/api/config/connection/", ConnectHandler, common_kwargs),
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


def main(host, port, numeric_level, event_registry, shared_state, token):
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    )
    log.info(f"listening on {host}:{port}")
    app = make_app(event_registry, shared_state, token)
    app.listen(address=host, port=port)
    return app


def _confirm_exit(shared_state, loop):
    log.info('interrupted')
    sys.stdout.write("Shutdown libertem server (y/[n])? ")
    sys.stdout.flush()
    r, w, x = select.select([sys.stdin], [], [], 5)
    if r:
        line = sys.stdin.readline()
        if line.lower().startswith('y'):
            log.critical("Shutdown confirmed")
            # schedule stop on main thread
            loop.add_callback_from_signal(
                lambda: asyncio.ensure_future(do_stop(shared_state))
            )
            return
    else:
        print('No answer for 5s: ')
    print('Resuming operation ...')
    # set it back to original SIGINT handler
    loop.add_callback_from_signal(partial(handle_signal, shared_state))


def _handle_exit(signum, frame, shared_state):
    loop = tornado.ioloop.IOLoop.current()
    # register more forceful signal handler for ^C^C case
    signal.signal(signal.SIGINT, partial(sig_exit, shared_state=shared_state))
    thread = threading.Thread(target=partial(_confirm_exit, shared_state, loop))
    thread.daemon = True
    thread.start()


def handle_signal(shared_state):
    if not sys.platform.startswith('win') and sys.stdin and sys.stdin.isatty():
        signal.signal(signal.SIGINT, partial(_handle_exit, shared_state=shared_state))
    else:
        signal.signal(signal.SIGINT, partial(sig_exit, shared_state=shared_state))


def run(host, port, browser, local_directory, numeric_level, token):
    # shared state:
    event_registry = EventRegistry()
    shared_state = SharedState()

    shared_state.set_local_directory(local_directory)
    main(host, port, numeric_level, event_registry, shared_state, token)
    if browser:
        webbrowser.open(f'http://{host}:{port}')
    loop = asyncio.get_event_loop()
    handle_signal(shared_state)
    # Strictly necessary only on Windows, but doesn't do harm in any case.
    # FIXME check later if the unknown root cause was fixed upstream
    asyncio.ensure_future(nannynanny())
    loop.run_forever()
