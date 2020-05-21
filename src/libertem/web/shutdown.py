import tornado
import logging
import asyncio

from .state import SharedState

log = logging.getLogger(__name__)


class ShutdownHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def delete(self):
        log.info("Handling shutdown button")
        if self.state.executor_state.executor is not None:
            await self.state.executor_state.executor.close()
        self.write({
            "status": "ok",
            "messageType": "SERVER_SHUTDOWN"
        })
        await self.flush()
        tornado.ioloop.IOLoop.current().stop()
