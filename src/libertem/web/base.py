import os
import time
import logging
import asyncio
from functools import partial

import tornado.web
import tornado.gen
import tornado.websocket
import tornado.ioloop
import tornado.escape
import psutil

import libertem
from .messages import Message
from libertem.executor.base import JobCancelledError
from libertem.io.dataset.base import DataSetException

log = logging.getLogger(__name__)


def log_message(message, exception=False):
    log_fn = log.info
    if exception:
        log_fn = log.exception
    if "job" in message:
        log_fn("message: %s (job=%s)" % (message["messageType"], message["job"]))
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


async def run_blocking(fn, *args, **kwargs):
    """
    run blocking function fn with args, kwargs in a thread and return a corresponding future
    """
    return await tornado.ioloop.IOLoop.current().run_in_executor(None, partial(fn, *args, **kwargs))


class RunJobMixin(object):
    async def run_job(self, uuid, ds, job, full_result):
        self.data.register_job(uuid=uuid, job=job)
        executor = self.data.get_executor()
        msg = Message(self.data).start_job(
            job_id=uuid,
        )
        log_message(msg)
        self.write(msg)
        self.finish()
        self.event_registry.broadcast_event(msg)

        t = time.time()
        try:
            async for result in executor.run_job(job):
                for tile in result:
                    tile.reduce_into_result(full_result)
                if time.time() - t < 0.3:
                    continue
                t = time.time()
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


class SharedData(object):
    def __init__(self):
        self.datasets = {}
        self.jobs = {}
        self.job_to_id = {}
        self.dataset_to_id = {}
        self.executor = None
        self.cluster_params = {}
        self.local_directory = "dask-worker-space"

    def get_local_cores(self, default=2):
        cores = psutil.cpu_count(logical=False)
        if cores is None:
            cores = default
        return cores

    def set_local_directory(self, local_directory):
        if local_directory is not None:
            self.local_directory = local_directory

    def get_local_directory(self):
        return self.local_directory

    def get_config(self):
        return {
            "version": libertem.__version__,
            "revision": libertem.revision,
            "localCores": self.get_local_cores(),
            "cwd": os.getcwd(),
            # '/' works on Windows, too.
            "separator": '/'
        }

    def get_executor(self):
        if self.executor is None:
            # TODO: exception type, conversion into 400 response
            raise RuntimeError("wrong state: executor is None")
        return self.executor

    def have_executor(self):
        return self.executor is not None

    def get_cluster_params(self):
        return self.cluster_params

    async def set_executor(self, executor, params):
        if self.executor is not None:
            await self.executor.close()
        self.executor = executor
        self.cluster_params = params

    def register_dataset(self, uuid, dataset, params):
        assert uuid not in self.datasets
        self.datasets[uuid] = {
            "dataset": dataset,
            "params": params,
        }
        self.dataset_to_id[dataset] = uuid
        return self

    def get_dataset(self, uuid):
        return self.datasets[uuid]["dataset"]

    async def verify_datasets(self):
        executor = self.get_executor()
        for uuid, params in self.datasets.items():
            dataset = params["dataset"]
            try:
                await executor.run_function(dataset.check_valid)
            except DataSetException:
                await self.remove_dataset(uuid)

    async def remove_dataset(self, uuid):
        ds = self.datasets[uuid]["dataset"]
        jobs_to_remove = [
            job
            for job in self.jobs.values()
            if self.dataset_to_id[job.dataset] == uuid
        ]
        job_ids = {self.job_to_id[job]
                   for job in jobs_to_remove}
        del self.datasets[uuid]
        del self.dataset_to_id[ds]
        for job_id in job_ids:
            await self.remove_job(job_id)

    def register_job(self, uuid, job):
        assert uuid not in self.jobs
        self.jobs[uuid] = job
        self.job_to_id[job] = uuid
        return self

    async def remove_job(self, uuid):
        try:
            job = self.jobs[uuid]
            executor = self.get_executor()
            await executor.cancel(job)
            del self.jobs[uuid]
            del self.job_to_id[job]
            return True
        except KeyError:
            return False

    def get_job(self, uuid):
        return self.jobs[uuid]

    def job_is_cancelled(self, uuid):
        return uuid not in self.jobs

    def serialize_job(self, job_id):
        job = self.jobs[job_id]
        return {
            "id": job_id,
            "dataset": self.dataset_to_id[job.dataset],
        }

    def serialize_jobs(self):
        return [
            self.serialize_job(job_id)
            for job_id in self.jobs.keys()
        ]

    async def serialize_dataset(self, dataset_id):
        executor = self.get_executor()
        dataset = self.datasets[dataset_id]
        diag = await executor.run_function(lambda: dataset["dataset"].diagnostics)
        return {
            "id": dataset_id,
            "params": {
                **dataset["params"]["params"],
                "shape": tuple(dataset["dataset"].shape),
            },
            "diagnostics": diag,
        }

    async def serialize_datasets(self):
        return [
            await self.serialize_dataset(dataset_id)
            for dataset_id in self.datasets.keys()
        ]
