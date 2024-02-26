import logging
from typing import TYPE_CHECKING

import tornado.web
from opentelemetry import trace

from libertem.web.engine import JobEngine
from libertem.executor.base import AsyncAdapter
from .messages import Message
from .base import log_message
from .state import SharedState

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

if TYPE_CHECKING:
    from libertem.web.events import EventRegistry


class ConnectHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry: "EventRegistry"):
        self.state = state
        self.event_registry = event_registry
        self.engine = JobEngine(state, event_registry)

    async def get(self):
        log.info("ConnectHandler.get")
        try:
            await self.state.executor_state.get_executor()
            params = self.state.executor_state.get_cluster_params()
            # TODO: extract into Message class
            self.write({
                "status": "ok",
                "connection": params["connection"],
            })
        except RuntimeError:  # TODO: exception class is too generic
            # TODO: extract into Message class
            self.write({
                "status": "disconnected",
                "connection": {},
            })

    async def put(self):
        with tracer.start_as_current_span("ConnectHandler.put"):
            await self._do_connect()

    async def _do_connect(self):
        # TODO: extract json request data stuff into mixin?
        request_data = tornado.escape.json_decode(self.request.body)
        pool = AsyncAdapter.make_pool()
        with tracer.start_as_current_span("executor setup"):
            try:
                executor = await self.state.executor_state.make_executor(request_data, pool)
            except Exception as e:
                msg = Message().cluster_conn_error(msg=str(e))
                log_message(msg, exception=True)
                self.set_status(500)
                self.write(msg)
                return None
            await self.state.executor_state.set_executor(executor, request_data)
        await self.state.dataset_state.verify()
        datasets = await self.state.dataset_state.serialize_all()
        msg = Message().initial_state(
            jobs=self.state.job_state.serialize_all(),
            datasets=datasets, analyses=self.state.analysis_state.serialize_all(),
            compound_analyses=self.state.compound_analysis_state.serialize_all(),
        )
        log_message(msg)
        # FIXME: don't broadcast, only send to the websocket that matches this HTTP connection
        # (is this even possible?)
        self.event_registry.broadcast_event(msg)
        await self.engine.send_existing_job_results()
        self.write({
            "status": "ok",
            "connection": request_data['connection'],
        })
