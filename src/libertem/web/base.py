import logging
import asyncio

from libertem.common.async_utils import sync_to_async
from libertem.analysis.base import AnalysisResultSet

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
    elif "msg" in message:
        log_fn("message: %s: %s", message["messageType"], message['msg'])
    else:
        log_fn("message: %s" % message["messageType"])


async def result_images(results: AnalysisResultSet, pool, save_kwargs=None):
    futures = [
        sync_to_async(result.get_image, pool, save_kwargs)
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
