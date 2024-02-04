import asyncio

import tornado.websocket

from libertem.web.engine import JobEngine

from .base import log_message
from .messages import Message
from .state import SharedState


class EventRegistry:
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)

    def remove_handler(self, handler):
        self.handlers.remove(handler)

    def broadcast_event(self, message, *args, **kwargs):
        futures = []
        for handler in self.handlers:
            try:
                future = handler.write_message(message, *args, **kwargs)
            except tornado.websocket.WebSocketClosedError:
                self.remove_handler(handler)
                continue
            futures.append(future)
        return asyncio.gather(*futures)

    def broadcast_together(self, messages, *args, **kwargs):
        futures = []
        for handler in self.handlers:
            for message in messages:
                futures.append(
                    handler.write_message(message, *args, **kwargs)
                )
        return asyncio.gather(*futures)


class ResultEventHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, state: SharedState, event_registry: EventRegistry):
        self.event_registry = event_registry
        self.state = state
        self.engine = JobEngine(state, event_registry)

    def check_origin(self, origin):
        # FIXME: implement this when we want to support CORS later
        return super().check_origin(origin)

    async def open(self):
        self.event_registry.add_handler(self)
        if self.state.executor_state.have_executor():
            await self.state.dataset_state.verify()
            datasets = await self.state.dataset_state.serialize_all()
            msg = Message().initial_state(
                jobs=self.state.job_state.serialize_all(),
                datasets=datasets, analyses=self.state.analysis_state.serialize_all(),
                compound_analyses=self.state.compound_analysis_state.serialize_all(),
            )
            log_message(msg)
            # FIXME: don't broadcast, only send to the new connection
            self.event_registry.broadcast_event(msg)
            await self.engine.send_existing_job_results()

    def on_close(self):
        self.event_registry.remove_handler(self)
