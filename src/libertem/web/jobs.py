import logging
from typing import TYPE_CHECKING

import tornado.web

from libertem.web.engine import JobEngine
from .base import CORSMixin, log_message
from .state import SharedState
from .messages import Message


if TYPE_CHECKING:
    from .events import EventRegistry

log = logging.getLogger(__name__)


class JobDetailHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry: "EventRegistry"):
        self.state = state
        self.event_registry = event_registry
        self.engine = JobEngine(state, event_registry)

    async def put(self, job_id):
        request_data = tornado.escape.json_decode(self.request.body)
        analysis_id = request_data['job']['analysis']
        await self.engine.register_job(analysis_id, job_id)
        msg = Message(self.state).start_job(
            job_id=job_id, analysis_id=analysis_id,
        )
        log_message(msg)
        self.write(msg)
        self.finish()
        await self.engine.run_analysis(analysis_id, job_id)

    async def delete(self, job_id):
        result = await self.state.job_state.remove(job_id)
        if result:
            msg = Message(self.state).cancel_job(job_id)
            log_message(msg)
            self.event_registry.broadcast_event(msg)
            self.write(msg)
        else:
            log.warning("tried to remove unknown job %s", job_id)
            msg = Message(self.state).cancel_failed(job_id)
            log_message(msg)
            self.event_registry.broadcast_event(msg)
            self.write(msg)
