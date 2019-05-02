import logging

import tornado.web
import tornado.gen
import tornado.websocket
import tornado.ioloop
import tornado.escape

from .base import log_message
from .messages import Message

log = logging.getLogger(__name__)


class ConfigHandler(tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    async def get(self):
        log.info("ConfigHandler.get")
        msg = Message(self.data).config(config=self.data.get_config())
        log_message(msg)
        self.write(msg)
