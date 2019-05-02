import tornado.websocket

from .base import log_message
from .messages import Message


class ResultEventHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, data, event_registry):
        self.registry = event_registry
        self.data = data

    def check_origin(self, origin):
        # FIXME: implement this when we want to support CORS later
        return super().check_origin(origin)

    async def open(self):
        self.registry.add_handler(self)
        if self.data.have_executor():
            await self.data.verify_datasets()
            datasets = await self.data.serialize_datasets()
            msg = Message(self.data).initial_state(
                jobs=self.data.serialize_jobs(),
                datasets=datasets,
            )
            log_message(msg)
            self.registry.broadcast_event(msg)

    def on_close(self):
        self.registry.remove_handler(self)


class EventRegistry(object):
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)

    def remove_handler(self, handler):
        self.handlers.remove(handler)

    def broadcast_event(self, message, *args, **kwargs):
        for handler in self.handlers:
            handler.write_message(message, *args, **kwargs)

    def broadcast_together(self, messages, *args, **kwargs):
        for handler in self.handlers:
            for message in messages:
                handler.write_message(message, *args, **kwargs)
