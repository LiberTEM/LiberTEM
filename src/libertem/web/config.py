import logging

import tornado.web
import tornado.gen
import tornado.websocket
import tornado.ioloop
import tornado.escape

from .base import log_message
from .messages import Message
from .state import SharedState

log = logging.getLogger(__name__)


class ConfigHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def get(self):
        log.info("ConfigHandler.get")
        msg = Message(self.state).config(config=self.state.get_config())
        log_message(msg)
        self.write(msg)


class ClusterDetailHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def get(self):
        executor = self.state.executor_state.get_executor()
        details = await executor.get_resource_details()
        msg = Message(self.state).cluster_details(details=details)
        log_message(msg)
        self.write(msg)
