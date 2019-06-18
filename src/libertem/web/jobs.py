import time
import logging

import tornado.web

from libertem.analysis import (
    DiskMaskAnalysis, RingMaskAnalysis, PointMaskAnalysis,
    COMAnalysis, SumAnalysis, PickFrameAnalysis, RadialFourierAnalysis
)
from .base import CORSMixin, run_blocking, log_message, result_images
from .messages import Message
from libertem.executor.base import JobCancelledError
from libertem.udf.base import UDFRunner

log = logging.getLogger(__name__)


class JobDetailHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    def get_analysis_by_type(self, type_):
        analysis_by_type = {
            "APPLY_DISK_MASK": DiskMaskAnalysis,
            "APPLY_RING_MASK": RingMaskAnalysis,
            "APPLY_POINT_SELECTOR": PointMaskAnalysis,
            "CENTER_OF_MASS": COMAnalysis,
            "RADIAL_FOURIER": RadialFourierAnalysis,
            "SUM_FRAMES": SumAnalysis,
            "PICK_FRAME": PickFrameAnalysis,
        }
        return analysis_by_type[type_]

    async def put(self, uuid):
        request_data = tornado.escape.json_decode(self.request.body)
        params = request_data['job']
        ds = self.data.get_dataset(params['dataset'])
        analysis = self.get_analysis_by_type(params['analysis']['type'])(
            dataset=ds,
            parameters=params['analysis']['parameters']
        )

        if analysis.TYPE == 'UDF':
            try:
                return await self.run_udf(uuid, ds, analysis)
            except Exception as e:
                log.exception("error running job, params=%r", params)
                msg = Message(self.data).job_error(uuid, "error running job: %s" % str(e))
                self.event_registry.broadcast_event(msg)
                await self.data.remove_job(uuid)
        else:
            job = analysis.get_job()
            full_result = job.get_result_buffer()
            job_runner = self.run_job(
                full_result=full_result,
                uuid=uuid, ds=ds, job=job,
            )
            try:
                await job_runner.asend(None)
                while True:
                    results = await run_blocking(
                        analysis.get_results,
                        job_results=full_result,
                    )
                    await job_runner.asend(results)
            except StopAsyncIteration:
                pass
            except Exception as e:
                log.exception("error running job, params=%r", params)
                msg = Message(self.data).job_error(uuid, "error running job: %s" % str(e))
                self.event_registry.broadcast_event(msg)
                await self.data.remove_job(uuid)

    async def delete(self, uuid):
        result = await self.data.remove_job(uuid)
        if result:
            msg = Message(self.data).cancel_job(uuid)
            log_message(msg)
            self.event_registry.broadcast_event(msg)
            self.write(msg)
        else:
            log.warning("tried to remove unknown job %s", uuid)
            msg = Message(self.data).cancel_failed(uuid)
            log_message(msg)
            self.event_registry.broadcast_event(msg)
            self.write(msg)

    async def run_udf(self, uuid, ds, analysis):
        udf = analysis.get_udf()
        roi = analysis.get_roi()

        # FIXME: register_job for UDFs?
        self.data.register_job(uuid=uuid, job=udf, dataset=ds)

        # FIXME: code duplication
        executor = self.data.get_executor()
        msg = Message(self.data).start_job(
            job_id=uuid,
        )
        log_message(msg)
        self.write(msg)
        self.finish()
        self.event_registry.broadcast_event(msg)

        t = time.time()
        post_t = time.time()
        prev = 0
        async for udf_results in UDFRunner(udf).run_for_dataset_async(ds, executor, roi):
            results = await run_blocking(
                analysis.get_udf_results,
                udf_results=udf_results,
            )
            if time.time() - t < min(max(0.3, 2*(t - post_t), prev * 0.8), 10):
                continue
            prev = 2*(t - post_t)
            post_t = time.time()
            images = await result_images(results)

            # NOTE: make sure the following broadcast_event messages are sent atomically!
            # (that is: keep the code below synchronous, and only send the messages
            # once the images have finished encoding, and then send all at once)
            msg = Message(self.data).task_result(
                job_id=uuid,
                num_images=len(results),
                image_descriptions=[
                    {"title": result.title, "desc": result.desc}
                    for result in results
                ],
            )
            log_message(msg)
            self.event_registry.broadcast_event(msg)
            for image in images:
                raw_bytes = image.read()
                self.event_registry.broadcast_event(raw_bytes, binary=True)
            # The broadcast might have taken quite some time due to
            # backpressure from the network
            t = time.time()

        if self.data.job_is_cancelled(uuid):
            return
        results = await run_blocking(
            analysis.get_udf_results,
            udf_results=udf_results,
        )
        images = await result_images(results)
        if self.data.job_is_cancelled(uuid):
            return
        msg = Message(self.data).finish_job(
            job_id=uuid,
            num_images=len(results),
            image_descriptions=[
                {"title": result.title, "desc": result.desc}
                for result in results
            ],
        )
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        for image in images:
            raw_bytes = image.read()
            self.event_registry.broadcast_event(raw_bytes, binary=True)

    async def run_job(self, uuid, ds, job, full_result):
        self.data.register_job(uuid=uuid, job=job, dataset=job.dataset)
        executor = self.data.get_executor()
        msg = Message(self.data).start_job(
            job_id=uuid,
        )
        log_message(msg)
        self.write(msg)
        self.finish()
        self.event_registry.broadcast_event(msg)

        t = time.time()
        post_t = time.time()
        prev = 0
        try:
            async for result in executor.run_job(job):
                for tile in result:
                    tile.reduce_into_result(full_result)
                if time.time() - t < min(max(0.3, 2*(t - post_t), prev * 0.8), 10):
                    continue
                prev = 2*(t - post_t)
                post_t = time.time()
                results = yield full_result
                images = await result_images(results)

                # NOTE: make sure the following broadcast_event messages are sent atomically!
                # (that is: keep the code below synchronous, and only send the messages
                # once the images have finished encoding, and then send all at once)
                msg = Message(self.data).task_result(
                    job_id=uuid,
                    num_images=len(results),
                    image_descriptions=[
                        {"title": result.title, "desc": result.desc}
                        for result in results
                    ],
                )
                log_message(msg)
                self.event_registry.broadcast_event(msg)
                for image in images:
                    raw_bytes = image.read()
                    self.event_registry.broadcast_event(raw_bytes, binary=True)
                # The broadcast might have taken quite some time due to
                # backpressure from the network
                t = time.time()
        except JobCancelledError:
            return  # TODO: maybe write a message on the websocket?

        results = yield full_result
        if self.data.job_is_cancelled(uuid):
            return
        images = await result_images(results)
        if self.data.job_is_cancelled(uuid):
            return
        msg = Message(self.data).finish_job(
            job_id=uuid,
            num_images=len(results),
            image_descriptions=[
                {"title": result.title, "desc": result.desc}
                for result in results
            ],
        )
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        for image in images:
            raw_bytes = image.read()
            self.event_registry.broadcast_event(raw_bytes, binary=True)
