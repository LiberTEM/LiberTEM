import logging

import tornado.escape
import tornado.gen
import tornado.ioloop
import tornado.web
import tornado.websocket

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
