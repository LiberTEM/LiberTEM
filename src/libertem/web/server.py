import os
import sys
import logging
import asyncio
import signal
import socket
import select
import threading
import webbrowser
import ipaddress
import urllib.parse
from functools import partial
from typing import Optional
import hmac
import hashlib
import time

import tornado.web
import tornado.gen
import tornado.websocket
import tornado.ioloop
import tornado.escape
from tornado import httputil

from ..common.async_utils import adjust_event_loop_policy
from .shutdown import ShutdownHandler
from .state import SharedState, ExecutorState
from .config import ConfigHandler, ClusterDetailHandler
from .dataset import DataSetDetailHandler, DataSetDetectHandler
from .browse import LocalFSBrowseHandler, LocalFSStatHandler
from .jobs import JobDetailHandler
from .events import ResultEventHandler, EventRegistry
from .connect import ConnectHandler
from .analysis import (
    AnalysisDetailHandler, DownloadDetailHandler, CompoundAnalysisHandler,
    AnalysisRPCHandler,
)
from .generator import DownloadScriptHandler, CopyScriptHandler
from .event_bus import EventBus, MessagePump

log = logging.getLogger(__name__)


class IndexHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    def get(self):
        self.render("client/index.html")


def is_localhost(host):
    return host in ['localhost', '127.0.0.1', '::1']


def _get_token(request):
    token = request.query_arguments.get('token', [b''])[0].decode("utf-8")
    if not token:
        token = request.headers.get('X-Api-Key', '')
    return token


def _write_token_mismatch_error(request):
    start_line = httputil.ResponseStartLine("", 400, "bad request: token mismatch")
    headers = httputil.HTTPHeaders({
        "Content-Type": "text/html; charset=UTF-8",
        "Date": httputil.format_timestamp(time.time()),
    })
    return request.connection.write_headers(
        start_line, headers, b""
    )


def check_token_auth_middleware(app, expected_token):
    # bypass token check if no token is set:
    if expected_token is None:
        return lambda r: app(r)

    expected_hash = hashlib.sha256(expected_token.encode("utf8")).hexdigest()

    def _middleware(request):
        # NOTE: token length may be leaked here
        given_token = _get_token(request)
        given_hash = hashlib.sha256(given_token.encode("utf8")).hexdigest()
        if not hmac.compare_digest(given_hash, expected_hash):
            return _write_token_mismatch_error(request)
        return app(request)

    return _middleware


def make_app(event_registry, shared_state, token=None):
    """
    Returns the fully assembled web API app, which is a
    callable(request) (not the "raw" tornado Application
    instance, because we want to apply middleare around it)
    """
    settings = {
        "static_path": os.path.join(os.path.dirname(__file__), "client"),
    }
    assets_path = os.path.join(os.path.dirname(__file__), "client", "assets")
    common_kwargs = {
        "state": shared_state,
        "event_registry": event_registry,
    }
    app = tornado.web.Application([
        (r"/", IndexHandler, common_kwargs),
        (r"/api/datasets/detect/", DataSetDetectHandler, common_kwargs),
        (r"/api/datasets/([^/]+)/", DataSetDetailHandler, common_kwargs),
        (r"/api/browse/localfs/", LocalFSBrowseHandler, common_kwargs),
        (r"/api/browse/localfs/stat/", LocalFSStatHandler, common_kwargs),
        (r"/api/jobs/([^/]+)/", JobDetailHandler, common_kwargs),
        (r"/api/compoundAnalyses/([^/]+)/analyses/([^/]+)/", AnalysisDetailHandler, common_kwargs),
        (
            r"/api/compoundAnalyses/([^/]+)/analyses/([^/]+)/download/([^/]+)/",
            DownloadDetailHandler,
            common_kwargs
        ),
        (r"/api/compoundAnalyses/([^/]+)/rpc/([^/]+)/", AnalysisRPCHandler, common_kwargs),
        (r"/api/compoundAnalyses/([^/]+)/copy/notebook/", CopyScriptHandler, common_kwargs),
        (r"/api/compoundAnalyses/([^/]+)/download/notebook/", DownloadScriptHandler, common_kwargs),
        (r"/api/compoundAnalyses/([^/]+)/", CompoundAnalysisHandler, common_kwargs),
        (r"/api/events/", ResultEventHandler, common_kwargs),
        (r"/api/shutdown/", ShutdownHandler, common_kwargs),
        (r"/api/config/", ConfigHandler, common_kwargs),
        (r"/api/config/cluster/", ClusterDetailHandler, common_kwargs),
        (r"/api/config/connection/", ConnectHandler, common_kwargs),
        (r"/assets/(.*)", tornado.web.StaticFileHandler, {"path": assets_path}),
    ], **settings)
    app = check_token_auth_middleware(app, token)
    return app


async def do_stop(shared_state):
    log.warning("Exiting...")
    shared_state.executor_state.shutdown()
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


def main(bound_sockets, event_registry, shared_state, token):
    app = make_app(event_registry, shared_state, token)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.add_sockets(bound_sockets)
    return http_server


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


def port_from_sockets(*sockets):
    ports = tuple(s.getsockname()[1] for s in sockets)
    assert ports, 'No sockets'
    return ports[0]


def run(
    host, port, browser, local_directory, numeric_level,
    token, preload, strict_port, executor_spec, open_ds,
    snooze_timeout: Optional[float] = None,
):
    logging.basicConfig(
        level=numeric_level,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    )

    adjust_event_loop_policy()
    loop = asyncio.get_event_loop()

    # shared state:
    event_bus = EventBus()
    event_registry = EventRegistry()
    pump = MessagePump(event_bus=event_bus, event_registry=event_registry)
    executor_state = ExecutorState(event_bus=event_bus, snooze_timeout=snooze_timeout, loop=loop)
    shared_state = SharedState(executor_state=executor_state)

    executor_state.set_local_directory(local_directory)
    executor_state.set_preload(preload)

    try:
        bound_sockets = tornado.netutil.bind_sockets(port, host)
    except OSError as e:
        if strict_port:
            raise e
        bound_sockets = tornado.netutil.bind_sockets(0, host)
        _port = port_from_sockets(*bound_sockets)
        log.info(f"port {port} already in use, using random open port {_port}")
        port = _port

    try:
        main(bound_sockets, event_registry, shared_state, token)

        async def create_and_set_executor():
            if executor_spec is not None:
                await shared_state.create_and_set_executor(executor_spec)

        try:
            is_ipv6 = isinstance(ipaddress.ip_address(host), ipaddress.IPv6Address)
        except ValueError:
            is_ipv6 = False
        url = f'http://[{host}]:{port}' if is_ipv6 else f'http://{host}:{port}'
        if open_ds is not None:
            url = f'{url}/#action=open&path={open_ds}'
        msg = f"""

    LiberTEM listening on {url}"""
        parts = urllib.parse.urlsplit(url)
        if parts.hostname in ('0.0.0.0', '::'):
            hostname = socket.gethostname()
            mod_url = parts._replace(netloc=f'{hostname}:{parts.port}')
            msg = msg + f"""
                        {urllib.parse.urlunsplit(mod_url)}
    """
        else:
            # For display consistency
            msg = msg + "\n"
        log.info(msg)
        if browser:
            webbrowser.open(url)
        handle_signal(shared_state)
        asyncio.ensure_future(pump.run())
        # Strictly necessary only on Windows, but doesn't do harm in any case.
        # FIXME check later if the unknown root cause was fixed upstream
        asyncio.ensure_future(nannynanny())
        asyncio.ensure_future(create_and_set_executor())
        loop.run_forever()
    finally:
        executor_state.shutdown()
