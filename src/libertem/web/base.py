import logging
import asyncio
import hmac
import hashlib

from tornado.web import HTTPError
from libertem.utils.async_utils import sync_to_async
from libertem.executor.base import JobCancelledError
from libertem.analysis.base import AnalysisResultSet
from .messages import Message

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


class TokenAuthMixin:
    def check_token(self):
        if self.token is not None:
            # NOTE: token length may be leaked here
            given_token = self.get_query_argument('token', '')
            given_hash = hashlib.sha256(given_token.encode("utf8")).hexdigest()
            expected_hash = hashlib.sha256(self.token.encode("utf8")).hexdigest()
            if not hmac.compare_digest(given_hash, expected_hash):
                raise HTTPError(status_code=400, log_message="token mismatch")

    async def prepare(self):
        self.check_token()


class ResultHandlerMixin:
    async def send_results(self, results: AnalysisResultSet, job_id, analysis_id,
                           details, finished=False):
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
            self.state.analysis_state.set_results(analysis_id, details, results, job_id)
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
        for analysis_id, (details, result_set, job_id) in results:
            await self.event_registry.broadcast_event(
                Message(self.state).start_job(
                    job_id=job_id, analysis_id=analysis_id,
                )
            )
            await self.send_results(
                result_set, job_id, analysis_id, details, finished=True
            )
