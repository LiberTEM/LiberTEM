import time
import logging
import asyncio

import tornado.web

from libertem.analysis import (
    DiskMaskAnalysis, RingMaskAnalysis, PointMaskAnalysis,
    FEMAnalysis, COMAnalysis, SumAnalysis, PickFrameAnalysis,
    PickFFTFrameAnalysis, SumfftAnalysis,
    RadialFourierAnalysis, ApplyFFTMask, SDAnalysis, SumSigAnalysis, ClusterAnalysis
)
from .base import CORSMixin, log_message, result_images
from .state import SharedState
from .messages import Message
from libertem.executor.base import JobCancelledError
from libertem.udf.base import UDFRunner
from libertem.utils.async_utils import run_blocking

log = logging.getLogger(__name__)


class JobDetailHandler(CORSMixin, tornado.web.RequestHandler):
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

    async def put(self, uuid):
        request_data = tornado.escape.json_decode(self.request.body)
        analysis_id = request_data['job']['analysis']
        analysis_details = self.state.get_analysis(analysis_id)

        analysis_type = analysis_details["type"]
        params = analysis_details["params"]
        ds = self.state.get_dataset(analysis_details['dataset'])

        analysis = self.get_analysis_by_type(analysis_type)(
            dataset=ds,
            parameters=params,
        )

        try:
            assert analysis.TYPE == 'UDF'
            return await self.run_udf(uuid, ds, analysis)
        except JobCancelledError:
            msg = Message(self.state).cancel_done(uuid)
            log_message(msg)
            await self.event_registry.broadcast_event(msg)
            return
        except Exception as e:
            log.exception("error running job, params=%r", params)
            msg = Message(self.state).job_error(uuid, "error running job: %s" % str(e))
            self.event_registry.broadcast_event(msg)
            await self.state.remove_job(uuid)

    async def delete(self, uuid):
        result = await self.state.remove_job(uuid)
        if result:
            msg = Message(self.state).cancel_job(uuid)
            log_message(msg)
            self.event_registry.broadcast_event(msg)
            self.write(msg)
        else:
            log.warning("tried to remove unknown job %s", uuid)
            msg = Message(self.state).cancel_failed(uuid)
            log_message(msg)
            self.event_registry.broadcast_event(msg)
            self.write(msg)

    async def run_udf(self, uuid, ds, analysis):
        udf = analysis.get_udf()
        roi = analysis.get_roi()

        # FIXME: register_job for UDFs?
        self.state.register_job(
            uuid=uuid, job=udf, dataset=ds,
        )

        # FIXME: code duplication
        executor = self.state.get_executor()
        msg = Message(self.state).start_job(
            job_id=uuid,
        )
        log_message(msg)
        self.write(msg)
        self.finish()
        self.event_registry.broadcast_event(msg)

        if hasattr(analysis, 'controller'):
            return await analysis.controller(
                cancel_id=uuid, executor=executor,
                job_is_cancelled=lambda: self.state.job_is_cancelled(uuid),
                send_results=lambda results, finished: self.send_results(results, uuid,
                finished=finished)
            )

        t = time.time()
        post_t = time.time()
        window = 0.3
        result_iter = UDFRunner(udf).run_for_dataset_async(
            ds, executor, roi=roi, cancel_id=uuid
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
            await self.send_results(results, uuid)
            # The broadcast might have taken quite some time due to
            # backpressure from the network
            t = time.time()

        if self.state.job_is_cancelled(uuid):
            raise JobCancelledError()
        results = await run_blocking(
            analysis.get_udf_results,
            udf_results=udf_results,
            roi=roi,
        )
        await self.send_results(results, uuid, finished=True)

    async def send_results(self, results, uuid, finished=False):
        if self.state.job_is_cancelled(uuid):
            raise JobCancelledError()
        images = await result_images(results)
        if self.state.job_is_cancelled(uuid):
            raise JobCancelledError()
        if finished:
            msg = Message(self.state).finish_job(
                job_id=uuid,
                num_images=len(results),
                image_descriptions=[
                    {"title": result.title, "desc": result.desc}
                    for result in results
                ],
            )
        else:
            msg = Message(self.state).task_result(
                job_id=uuid,
                num_images=len(results),
                image_descriptions=[
                    {"title": result.title, "desc": result.desc}
                    for result in results
                ],
            )
        log_message(msg)
        # NOTE: make sure the following broadcast_event messages are sent atomically!
        # (that is: keep the code below synchronous, and only send the messages
        # once the images have finished encoding, and then send all at once)
        futures = []
        futures.append(
            self.event_registry.broadcast_event(msg)
        )
        for image in images:
            raw_bytes = image.read()
            futures.append(
                self.event_registry.broadcast_event(raw_bytes, binary=True)
            )
        await asyncio.gather(*futures)
