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
        workers = await executor.get_available_workers()
        details = {}

        for worker in workers:
            if worker.name.startswith("tcp"):
                host_name = worker.host
                resource = 'cpu'
            else:
                host_name = '-'.join(worker.name.split('-')[:-2])
                resource = worker.name.split('-')[-2]

            if host_name not in details.keys():
                details[host_name] = {
                                 'host': host_name,
                                 'cpu': 0,
                                 'cuda': 0,
                                 'service': 0,
                            }
            details[host_name][resource] += 1
        details_sorted = []
        for host in sorted(details.keys()):
            details_sorted.append(details[host])
        msg = Message(self.state).cluster_details(details=details_sorted)
        log_message(msg)
        self.write(msg)
