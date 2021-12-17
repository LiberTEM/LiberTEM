import logging
from typing import TYPE_CHECKING
import time

import tornado.web

from libertem.web.engine import JobEngine
from .base import CORSMixin, log_message
from .state import SharedState
from .messages import Message
from libertem.executor.base import JobCancelledError
from libertem.utils.async_utils import sync_to_async


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

    async def run_udf(self, job_id, dataset, dataset_id, analysis, analysis_id, details):
        udf = analysis.get_udf()
        roi = analysis.get_roi()

        executor = self.state.executor_state.get_executor()
        msg = Message(self.state).start_job(
            job_id=job_id, analysis_id=analysis_id,
        )
        log_message(msg)
        self.write(msg)
        self.finish()
        self.event_registry.broadcast_event(msg)

        if hasattr(analysis, 'controller'):
            return await analysis.controller(
                cancel_id=job_id, executor=executor,
                job_is_cancelled=lambda: self.state.job_state.is_cancelled(job_id),
                send_results=lambda results, finished: self.send_results(
                    results, job_id, finished=finished,
                    details=details, analysis_id=analysis_id,
                )
            )

        # short circuit if the parameters only change the visualization
        # (as determined by the analysis via `Analysis.need_rerun`):
        if self.state.analysis_state.have_results(analysis_id):
            old_results = self.state.analysis_state.get_results(analysis_id)
            old_details, _, _, old_udf_results = old_results
            if not analysis.need_rerun(
                old_details["parameters"],
                details["parameters"],
            ):
                results = await sync_to_async(
                    analysis.get_udf_results,
                    udf_results=old_udf_results.buffers[0],
                    roi=roi,
                    damage=old_udf_results.damage
                )
                await self.send_results(
                    results, job_id, analysis_id, details, finished=True,
                    udf_results=old_udf_results,
                )
                return

        t = time.time()
        post_t = time.time()
        window = 0.3
        # FIXME: allow to set correction data for a dataset via upload and local loading
        corrections = dataset.get_correction_data()
        runner_cls = executor.get_udf_runner()
        result_iter = runner_cls([udf]).run_for_dataset_async(
            dataset, executor, roi=roi, cancel_id=job_id, corrections=corrections,
        )
        async for udf_results in result_iter:
            window = min(max(window, 2*(t - post_t)), 5)
            if time.time() - t < window:
                continue
            results = await sync_to_async(
                analysis.get_udf_results,
                udf_results=udf_results.buffers[0],
                roi=roi,
                damage=udf_results.damage
            )
            post_t = time.time()
            await self.send_results(
                results, job_id, analysis_id, details, udf_results=udf_results
            )
            # The broadcast might have taken quite some time due to
            # backpressure from the network
            t = time.time()

        if self.state.job_state.is_cancelled(job_id):
            raise JobCancelledError()
        results = await sync_to_async(
            analysis.get_udf_results,
            udf_results=udf_results.buffers[0],
            roi=roi,
            damage=udf_results.damage
        )
        await self.send_results(
            results, job_id, analysis_id, details, finished=True, udf_results=udf_results
        )
