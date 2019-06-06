import logging

import tornado.web

from libertem.analysis import (
    DiskMaskAnalysis, RingMaskAnalysis, PointMaskAnalysis,
    COMAnalysis, SumAnalysis, PickFrameAnalysis, SumfftAnalysis
)
from .base import CORSMixin, RunJobMixin, run_blocking, log_message
from .messages import Message

log = logging.getLogger(__name__)


class JobDetailHandler(CORSMixin, RunJobMixin, tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    def get_analysis_by_type(self, type_):
        analysis_by_type = {
            "APPLY_DISK_MASK": DiskMaskAnalysis,
            "APPLY_RING_MASK": RingMaskAnalysis,
            "FFTSUM_FRAMES": SumfftAnalysis,
            "APPLY_POINT_SELECTOR": PointMaskAnalysis,
            "CENTER_OF_MASS": COMAnalysis,
            "SUM_FRAMES": SumAnalysis,
            "PICK_FRAME": PickFrameAnalysis,
            "APPLY_FFT_MASK": RingMaskAnalysis,
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
