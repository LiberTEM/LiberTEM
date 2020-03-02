import logging
import asyncio

from libertem.utils.async_utils import run_blocking

log = logging.getLogger(__name__)


def log_message(message, exception=False):
    log_fn = log.info
    if exception:
        log_fn = log.exception
    if "job" in message:
        log_fn("message: %s (job=%s)" % (message["messageType"], message["job"]))
    elif "analysis" in message:
        log_fn("message: %s (analysis=%s)" % (message["messageType"], message["analysis"]))
    elif "dataset" in message:
        log_fn("message: %s (dataset=%s)" % (message["messageType"], message["dataset"]))
    else:
        log_fn("message: %s" % message["messageType"])


async def result_images(results, save_kwargs=None):
    futures = [
        run_blocking(result.get_image, save_kwargs)
        for result in results
    ]

    images = await asyncio.gather(*futures)
    return images


class CORSMixin(object):
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
