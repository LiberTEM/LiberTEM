import os
import time
import logging
import asyncio
import signal
import psutil
from functools import partial

import tornado.web
import tornado.gen
import tornado.websocket
import tornado.ioloop
import tornado.escape

import libertem
from libertem.io.fs import get_fs_listing, FSError
from libertem.executor.dask import DaskJobExecutor
from libertem.executor.base import JobCancelledError, AsyncAdapter, sync_to_async
from libertem.io.dataset.base import DataSetException
from libertem.io import dataset
from libertem.analysis import (
    DiskMaskAnalysis, RingMaskAnalysis, PointMaskAnalysis,
    COMAnalysis, SumAnalysis, PickFrameAnalysis
)


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


class Message(object):
    """
    possible messages - the translation of our python datatypes to json types
    """

    def __init__(self, data):
        self.data = data

    def initial_state(self, jobs, datasets):
        return {
            "status": "ok",
            "messageType": "INITIAL_STATE",
            "datasets": datasets,
            "jobs": jobs,
        }

    def config(self, config):
        return {
            "status": "ok",
            "messageType": "CONFIG",
            "config": config,
        }

    def create_dataset(self, dataset, details):
        return {
            "status": "ok",
            "messageType": "CREATE_DATASET",
            "dataset": dataset,
            "details": details,
        }

    def create_dataset_error(self, dataset, msg):
        return {
            "status": "error",
            "messageType": "CREATE_DATASET_ERROR",
            "dataset": dataset,
            "msg": msg,
        }

    def delete_dataset(self, dataset):
        return {
            "status": "ok",
            "messageType": "DELETE_DATASET",
            "dataset": dataset,
        }

    def dataset_detect(self, params):
        return {
            "status": "ok",
            "messageType": "DATASET_DETECTED",
            "datasetParams": params,
        }

    def dataset_detect_failed(self, path):
        return {
            "status": "error",
            "messageType": "DATASET_DETECTION_FAILED",
            "path": path,
            "msg": "could not automatically determine dataset format",
        }

    def start_job(self, job_id):
        return {
            "status": "ok",
            "messageType": "JOB_STARTED",
            "job": job_id,
            "details": self.data.serialize_job(job_id),
        }

    def job_error(self, job_id, msg):
        return {
            "status": "error",
            "messageType": "JOB_ERROR",
            "job": job_id,
            "msg": msg,
        }

    def finish_job(self, job_id, num_images, image_descriptions):
        return {
            "status": "ok",
            "messageType": "FINISH_JOB",
            "job": job_id,
            "details": self.data.serialize_job(job_id),
            "followup": {
                "numMessages": num_images,
                "descriptions": image_descriptions,
            },
        }

    def cancel_job(self, job_id):
        return {
            "status": "ok",
            "messageType": "CANCEL_JOB",
            "job": job_id,
        }

    def cancel_failed(self, job_id):
        return {
            "status": "error",
            "messageType": "CANCEL_JOB_FAILED",
            "job": job_id,
        }

    def task_result(self, job_id, num_images, image_descriptions):
        return {
            "status": "ok",
            "messageType": "TASK_RESULT",
            "job": job_id,
            "followup": {
                "numMessages": num_images,
                "descriptions": image_descriptions,
            },
        }

    def directory_listing(self, path, files, dirs, drives, places):
        def _details(item):
            return {
                "name":  item["name"],
                "size":  item["stat"].st_size,
                "ctime": item["stat"].st_ctime,
                "mtime": item["stat"].st_mtime,
                "owner": item["owner"],
            }

        return {
            "status": "ok",
            "messageType": "DIRECTORY_LISTING",
            "drives": drives,
            "places": places,
            "path": path,
            "files": [
                _details(f)
                for f in files
            ],
            "dirs": [
                _details(d)
                for d in dirs
            ],
        }

    def browse_failed(self, path, code, msg, alternative=None):
        return {
            "status": "error",
            "messageType": "DIRECTORY_LISTING_FAILED",
            "path": path,
            "code": code,
            "msg": msg,
            "alternative": alternative,
        }


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


class ResultEventHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, data, event_registry):
        self.registry = event_registry
        self.data = data

    def check_origin(self, origin):
        # FIXME: implement this when we want to support CORS later
        return super().check_origin(origin)

    async def open(self):
        self.registry.add_handler(self)
        if self.data.have_executor():
            await self.data.verify_datasets()
            datasets = await self.data.serialize_datasets()
            msg = Message(self.data).initial_state(
                jobs=self.data.serialize_jobs(),
                datasets=datasets,
            )
            log_message(msg)
            self.registry.broadcast_event(msg)

    def on_close(self):
        self.registry.remove_handler(self)


class EventRegistry(object):
    def __init__(self):
        self.handlers = []

    def add_handler(self, handler):
        self.handlers.append(handler)

    def remove_handler(self, handler):
        self.handlers.remove(handler)

    def broadcast_event(self, message, *args, **kwargs):
        for handler in self.handlers:
            handler.write_message(message, *args, **kwargs)

    def broadcast_together(self, messages, *args, **kwargs):
        for handler in self.handlers:
            for message in messages:
                handler.write_message(message, *args, **kwargs)


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


class JobDetailHandler(CORSMixin, RunJobMixin, tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    def get_analysis_by_type(self, type_):
        analysis_by_type = {
            "APPLY_DISK_MASK": DiskMaskAnalysis,
            "APPLY_RING_MASK": RingMaskAnalysis,
            "APPLY_POINT_SELECTOR": PointMaskAnalysis,
            "CENTER_OF_MASS": COMAnalysis,
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


class DataSetDetailHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    async def delete(self, uuid):
        try:
            self.data.get_dataset(uuid)
        except KeyError:
            self.set_status(404, "dataset with uuid %s not found" % uuid)
            return
        await self.data.remove_dataset(uuid)
        msg = Message(self.data).delete_dataset(uuid)
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)

    async def put(self, uuid):
        request_data = tornado.escape.json_decode(self.request.body)
        params = request_data['dataset']['params']
        # TODO: validate request_data
        # let's start simple:
        assert params['type'].lower() in ["hdfs", "hdf5", "raw", "mib", "blo", "k2is", "ser",
                                          "frms6"]
        if params["type"].lower() == "hdfs":
            dataset_params = {
                "index_path": params["path"],
                "tileshape": params["tileshape"],
                "host": "localhost",  # FIXME: config param
                "port": 8020,  # FIXME: config param
            }
        elif params["type"].lower() == "hdf5":
            dataset_params = {
                "path": params["path"],
                "ds_path": params["ds_path"],
                "tileshape": params["tileshape"],
            }
        elif params["type"].lower() == "raw":
            dataset_params = {
                "path": params["path"],
                "dtype": params["dtype"],
                "detector_size_raw": params["detector_size_raw"],
                "crop_detector_to": params["crop_detector_to"],
                "tileshape": params["tileshape"],
                "scan_size": params["scan_size"],
            }
        elif params["type"].lower() == "mib":
            dataset_params = {
                "path": params["path"],
                "tileshape": params["tileshape"],
                "scan_size": params["scan_size"],
            }
        elif params["type"].lower() == "blo":
            dataset_params = {
                "path": params["path"],
                "tileshape": params["tileshape"],
            }
        elif params["type"].lower() == "k2is":
            dataset_params = {
                "path": params["path"],
            }
        elif params["type"].lower() == "ser":
            dataset_params = {
                "path": params["path"],
            }
        elif params["type"].lower() == "frms6":
            dataset_params = {
                "path": params["path"],
            }
        try:
            executor = self.data.get_executor()
            ds = await executor.run_function(dataset.load,
                                             filetype=params["type"], **dataset_params)
            ds = await executor.run_function(ds.initialize)
            ds.set_num_cores(len(executor.get_available_workers()))
            await executor.run_function(ds.check_valid)
            self.data.register_dataset(
                uuid=uuid,
                dataset=ds,
                params=request_data['dataset'],
            )
            details = await self.data.serialize_dataset(dataset_id=uuid)
            msg = Message(self.data).create_dataset(dataset=uuid, details=details)
            log_message(msg)
            self.write(msg)
            self.event_registry.broadcast_event(msg)
        except Exception as e:
            msg = Message(self.data).create_dataset_error(uuid, str(e))
            log_message(msg, exception=True)
            self.write(msg)
            return


class DataSetDetectHandler(tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    async def get(self):
        path = self.request.arguments['path'][0].decode("utf8")
        executor = self.data.get_executor()

        params = await executor.run_function(dataset.detect, path=path)
        if not params:
            msg = Message(self.data).dataset_detect_failed(path=path)
            log_message(msg)
            self.write(msg)
            return
        params['type'] = params['type'].upper()
        msg = Message(self.data).dataset_detect(params=params)
        log_message(msg)
        self.write(msg)


class SharedData(object):
    def __init__(self):
        self.datasets = {}
        self.jobs = {}
        self.job_to_id = {}
        self.dataset_to_id = {}
        self.executor = None
        self.cluster_params = {}

    def get_local_cores(self, default=2):
        cores = psutil.cpu_count(logical=False)
        if cores is None:
            cores = default
        return cores

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


class ConfigHandler(tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    async def get(self):
        log.info("ConfigHandler.get")
        msg = Message(self.data).config(config=self.data.get_config())
        log_message(msg)
        self.write(msg)


class ConnectHandler(tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    async def get(self):
        log.info("ConnectHandler.get")
        try:
            self.data.get_executor()
            params = self.data.get_cluster_params()
            # TODO: extract into Message class
            self.write({
                "status": "ok",
                "connection": params["connection"],
            })
        except RuntimeError:  # TODO: exception class is too generic
            # TODO: extract into Message class
            self.write({
                "status": "disconnected",
                "connection": {},
            })

    async def put(self):
        # TODO: extract json request data stuff into mixin?
        request_data = tornado.escape.json_decode(self.request.body)
        connection = request_data['connection']
        if connection["type"].lower() == "tcp":
            sync_executor = await sync_to_async(partial(DaskJobExecutor.connect,
                scheduler_uri=connection['address'],
            ))
        elif connection["type"].lower() == "local":
            cluster_kwargs = {
                "threads_per_worker": 1,
            }
            if "numWorkers" in connection:
                cluster_kwargs.update({"n_workers": connection["numWorkers"]})
            sync_executor = await sync_to_async(
                partial(DaskJobExecutor.make_local, cluster_kwargs=cluster_kwargs)
            )
        else:
            raise ValueError("unknown connection type")
        executor = AsyncAdapter(wrapped=sync_executor)
        await self.data.set_executor(executor, request_data)
        await self.data.verify_datasets()
        datasets = await self.data.serialize_datasets()
        msg = Message(self.data).initial_state(
            jobs=self.data.serialize_jobs(),
            datasets=datasets,
        )
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write({
            "status": "ok",
            "connection": connection,
        })


class IndexHandler(tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    def get(self):
        self.render("client/index.html")


class LocalFSBrowseHandler(tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    async def get(self):
        executor = self.data.get_executor()
        path = self.request.arguments['path']
        assert len(path) == 1
        path = path[0].decode("utf8")
        try:
            listing = await executor.run_function(get_fs_listing, path)
            msg = Message(self.data).directory_listing(
                **listing
            )
            self.write(msg)
        except FSError as e:
            msg = Message(self.data).browse_failed(
                path=path,
                code=e.code,
                msg=str(e),
                alternative=e.alternative,
            )
            self.write(msg)


def make_app(event_registry, shared_data):
    settings = {
        "static_path": os.path.join(os.path.dirname(__file__), "client"),
    }
    return tornado.web.Application([
        (r"/", IndexHandler, {"data": shared_data, "event_registry": event_registry}),
        (r"/api/datasets/detect/", DataSetDetectHandler, {
            "data": shared_data,
            "event_registry": event_registry
        }),
        (r"/api/datasets/([^/]+)/", DataSetDetailHandler, {
            "data": shared_data,
            "event_registry": event_registry
        }),
        (r"/api/browse/localfs/", LocalFSBrowseHandler, {
            "data": shared_data,
            "event_registry": event_registry
        }),
        (r"/api/jobs/([^/]+)/", JobDetailHandler, {
            "data": shared_data,
            "event_registry": event_registry
        }),
        (r"/api/events/", ResultEventHandler, {
            "data": shared_data,
            "event_registry": event_registry
        }),
        (r"/api/config/", ConfigHandler, {
            "data": shared_data,
            "event_registry": event_registry
        }),
        (r"/api/config/connection/", ConnectHandler, {
            "data": shared_data,
            "event_registry": event_registry,
        }),
    ], **settings)


async def do_stop(shared_data):
    log.warning("Exiting...")
    log.debug("closing executor")
    if shared_data.executor is not None:
        await shared_data.executor.close()
    loop = asyncio.get_event_loop()
    log.debug("shutting down async generators")
    await loop.shutdown_asyncgens()
    log.debug("stopping event loop")
    loop.stop()


def sig_exit(signum, frame, shared_data):
    loop = tornado.ioloop.IOLoop.instance()
    loop.add_callback_from_signal(
        lambda: asyncio.ensure_future(do_stop(shared_data))
    )


def main(host, port, event_registry, shared_data):
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    )
    log.info("listening on %s:%s" % (host, port))
    app = make_app(event_registry, shared_data)
    app.listen(address=host, port=port)
    return app


def run(host, port):
    # shared state:
    event_registry = EventRegistry()
    shared_data = SharedData()

    main(host, port, event_registry, shared_data)
    loop = asyncio.get_event_loop()
    signal.signal(signal.SIGINT, partial(sig_exit, shared_data=shared_data))
    loop.run_forever()


if __name__ == "__main__":
    main("0.0.0.0", 9000)
