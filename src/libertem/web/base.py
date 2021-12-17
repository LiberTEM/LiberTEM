import logging
import asyncio
from typing import Optional, TYPE_CHECKING
from typing_extensions import Protocol
from libertem.udf.base import UDFResults

from libertem.utils.async_utils import sync_to_async
from libertem.executor.base import JobCancelledError
from libertem.analysis.base import AnalysisResultSet
from libertem.web.models import AnalysisDetails
from .messages import Message

if TYPE_CHECKING:
    from libertem.web.state import SharedState
    from libertem.web.events import EventRegistry

log = logging.getLogger(__name__)


def log_message(message, exception=False):
    log_fn = log.info
    if exception:
        log_fn = log.exception
    if "job" in message:
        log_fn("message: {} (job={})".format(message["messageType"], message["job"]))
    elif "analysis" in message:
        log_fn("message: {} (analysis={})".format(message["messageType"], message["analysis"]))
    elif "dataset" in message:
        log_fn("message: {} (dataset={})".format(message["messageType"], message["dataset"]))
    else:
        log_fn("message: %s" % message["messageType"])


async def result_images(results, save_kwargs=None):
    futures = [
        sync_to_async(result.get_image, save_kwargs)
        for result in results
    ]

    images = await asyncio.gather(*futures)
    return images


class CORSMixin:
    pass
    # FIXME: implement these when we want to support CORS later
#    def set_default_headers(self):
#        self.set_header("Access-Control-Allow-Origin", "*")  # XXX FIXME TODO!!!
#        # self.set_header("Access-Control-Allow-Headers", "x-requested-with")
#        self.set_header('Access-Control-Allow-Methods', 'PUT, POST, GET, OPTIONS')
#
#    def options(self, *args):
#        """
#        for CORS pre-flight requests, no body returned
#        """
#        self.set_status(204)
#        self.finish()


class ResultHandlerBase(Protocol):
    """
    The interface that any class using `ResultHandlerMixin` must implement.
    """
    state: "SharedState"
    event_registry: "EventRegistry"


class ResultHandlerMixin:
    async def send_results(
        self: ResultHandlerBase,
        results: AnalysisResultSet,
        job_id: str,
        analysis_id: str,
        details: AnalysisDetails,
        finished=False,
        udf_results: Optional[UDFResults] = None
    ) -> None:
        if self.state.job_state.is_cancelled(job_id):
            raise JobCancelledError()
        images = await result_images(results)
        if self.state.job_state.is_cancelled(job_id):
            raise JobCancelledError()
        if finished:
            msg = Message(self.state).finish_job(
                job_id=job_id,
                num_images=len(results),
                image_descriptions=[
                    {"title": result.title, "desc": result.desc,
                    "includeInDownload": result.include_in_download}
                    for result in results
                ],
            )
            self.state.analysis_state.set_results(
                analysis_id, details, results, job_id, udf_results
            )
        else:
            msg = Message(self.state).task_result(
                job_id=job_id,
                num_images=len(results),
                image_descriptions=[
                    {"title": result.title, "desc": result.desc,
                    "includeInDownload": result.include_in_download}
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

    async def send_existing_job_results(self):
        results = self.state.analysis_state.get_all_results()
        for analysis_id, (details, result_set, job_id, udf_results) in results:
            await self.event_registry.broadcast_event(
                Message(self.state).start_job(
                    job_id=job_id, analysis_id=analysis_id,
                )
            )
            await self.send_results(
                result_set, job_id, analysis_id, details, finished=True, udf_results=udf_results
            )
