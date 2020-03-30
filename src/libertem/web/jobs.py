import time
import logging

import tornado.web

from libertem.analysis import (
    DiskMaskAnalysis, RingMaskAnalysis, PointMaskAnalysis,
    FEMAnalysis, COMAnalysis, SumAnalysis, PickFrameAnalysis,
    PickFFTFrameAnalysis, SumfftAnalysis,
    RadialFourierAnalysis, ApplyFFTMask, SDAnalysis, SumSigAnalysis, ClusterAnalysis
)
from .base import CORSMixin, log_message, ResultHandlerMixin
from .state import SharedState
from .messages import Message
from libertem.executor.base import JobCancelledError
from libertem.udf.base import UDFRunner
from libertem.utils.async_utils import run_blocking

log = logging.getLogger(__name__)


class JobDetailHandler(CORSMixin, ResultHandlerMixin, tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    def get_analysis_by_type(self, type_):
        analysis_by_type = {
            "APPLY_DISK_MASK": DiskMaskAnalysis,
            "APPLY_RING_MASK": RingMaskAnalysis,
            "FFTSUM_FRAMES": SumfftAnalysis,
            "APPLY_POINT_SELECTOR": PointMaskAnalysis,
            "CENTER_OF_MASS": COMAnalysis,
            "RADIAL_FOURIER": RadialFourierAnalysis,
            "SUM_FRAMES": SumAnalysis,
            "PICK_FRAME": PickFrameAnalysis,
            "FEM": FEMAnalysis,
            "PICK_FFT_FRAME": PickFFTFrameAnalysis,
            "APPLY_FFT_MASK": ApplyFFTMask,
            "SD_FRAMES": SDAnalysis,
            "SUM_SIG": SumSigAnalysis,
            "CLUST": ClusterAnalysis
        }
        return analysis_by_type[type_]

    async def put(self, job_id):
        request_data = tornado.escape.json_decode(self.request.body)
        analysis_id = request_data['job']['analysis']
        analysis_state = self.state.analysis_state[analysis_id]
        ds = self.state.dataset_state[analysis_state['dataset']]

        analysis_details = analysis_state["details"]
        analysis_type = analysis_details["analysisType"]
        params = analysis_details["parameters"]

        analysis = self.get_analysis_by_type(analysis_type)(
            dataset=ds,
            parameters=params,
        )

        try:
            if analysis.TYPE != 'UDF':
                raise TypeError(
                    'Only Analysis classes with TYPE="UDF" are supported'
                )
            # FIXME: naming? job_state for UDFs?
            self.state.job_state.register(
                job_id=job_id, analysis_id=analysis_id, dataset_id=analysis_state['dataset'],
            )
            self.state.analysis_state.add_job(analysis_id, job_id)
            return await self.run_udf(
                job_id=job_id,
                dataset=ds,
                dataset_id=analysis_state['dataset'],
                analysis=analysis,
                analysis_id=analysis_id,
                details=analysis_details,
            )
        except JobCancelledError:
            msg = Message(self.state).cancel_done(job_id)
            log_message(msg)
            await self.event_registry.broadcast_event(msg)
            return
        except Exception as e:
            log.exception("error running job, params=%r", params)
            msg = Message(self.state).job_error(job_id, "error running job: %s" % str(e))
            self.event_registry.broadcast_event(msg)
            await self.state.job_state.remove(job_id)

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

        t = time.time()
        post_t = time.time()
        window = 0.3
        result_iter = UDFRunner(udf).run_for_dataset_async(
            dataset, executor, roi=roi, cancel_id=job_id,
        )
        async for udf_results in result_iter:
            window = min(max(window, 2*(t - post_t)), 5)
            if time.time() - t < window:
                continue
            results = await run_blocking(
                analysis.get_udf_results,
                udf_results=udf_results,
                roi=roi,
            )
            post_t = time.time()
            await self.send_results(results, job_id, analysis_id, details)
            # The broadcast might have taken quite some time due to
            # backpressure from the network
            t = time.time()

        if self.state.job_state.is_cancelled(job_id):
            raise JobCancelledError()
        results = await run_blocking(
            analysis.get_udf_results,
            udf_results=udf_results,
            roi=roi,
        )
        await self.send_results(results, job_id, analysis_id, details, finished=True)
