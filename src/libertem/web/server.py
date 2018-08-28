import os
import sys
import stat
import time
import datetime
import logging
import asyncio
import signal
import psutil
from functools import partial

import numpy as np
from matplotlib import cm
from dask import distributed as dd
from distributed.asyncio import AioClient
import tornado.web
import tornado.gen
import tornado.websocket
import tornado.ioloop
import tornado.escape

import libertem
from libertem.executor.dask import DaskJobExecutor
from libertem.io.dataset.base import DataSetException
from libertem.io import dataset
from libertem.job.sum import SumFramesJob
from libertem.job.raw import PickFrameJob
from libertem.common.slice import Slice
from libertem.viz import visualize_simple, encode_image
from libertem.analysis import (
    DiskMaskAnalysis, RingMaskAnalysis, PointMaskAnalysis,
    COMAnalysis, SumAnalysis
)


log = logging.getLogger(__name__)


def log_message(message):
    if "job" in message:
        log.info("message: %s (job=%s)" % (message["messageType"], message["job"]))
    else:
        log.info("message: %s" % message["messageType"])


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

    def create_dataset(self, dataset):
        return {
            "status": "ok",
            "messageType": "CREATE_DATASET",
            "dataset": dataset,
            "details": self.data.serialize_dataset(dataset),
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

    def start_job(self, job_id):
        return {
            "status": "ok",
            "messageType": "START_JOB",
            "job": job_id,
            "details": self.data.serialize_job(job_id),
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

    def directory_listing(self, path, files, dirs):
        def _details(item):
            return {
                "name":  item["name"],
                "size":  item["stat"].st_size,
                "mtime": item["stat"].st_mtime,
            }

        return {
            "status": "ok",
            "messageType": "DIRECTORY_LISTING",
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


class RunJobMixin(object):
    async def run_job(self, uuid, ds, job, full_result):
        self.data.register_job(uuid=uuid, job=job)
        executor = self.data.get_executor()

        futures = []
        for task in job.get_tasks():
            submit_kwargs = {}
            futures.append(
                executor.client.submit(task, **submit_kwargs)
            )
        self.write(Message(self.data).start_job(
            job_id=uuid
        ))
        self.finish()
        msg = Message(self.data).start_job(
            job_id=uuid,
        )
        log_message(msg)
        self.event_registry.broadcast_event(msg)

        t = time.time()
        async for future, result in dd.as_completed(futures, with_results=True):
            # TODO:
            # + only send PNG of area that has changed (bounding box of all result tiles!)
            # + normalize each channel (per channel: keep running min/max, map data to [0, 1])
            # + if min/max changes, send whole channel (all results up to this point re-normalized)
            # + maybe saturate up to some point (20% over current max => keep current max) and send
            #   whole result image once finished
            # + maybe use visualization framework in-browser (example: GR)

            # TODO: update task_result message:
            # + send bbox for blitting

            for tile in result:
                tile.copy_to_result(full_result)
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
        results = yield full_result
        images = await result_images(results)
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
        self.data.verify_datasets()
        if self.data.have_executor():
            msg = Message(self.data).initial_state(
                jobs=self.data.serialize_jobs(),
                datasets=self.data.serialize_datasets(),
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

    async def delete(self, uuid):
        # TODO: implement this. maybe by setting a flag, or by having all the futures in a list
        # in shared data and calling cancel on them
        raise NotImplementedError()


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
        self.data.remove_dataset(uuid)
        msg = Message(self.data).delete_dataset(uuid)
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)

    async def put(self, uuid):
        request_data = tornado.escape.json_decode(self.request.body)
        params = request_data['dataset']['params']
        # TODO: validate request_data
        # let's start simple:
        assert params['type'].lower() in ["hdfs", "hdf5", "raw", "mib", "blo", "k2is"]
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
                "ds_path": params["dsPath"],
                "tileshape": params["tileshape"],
            }
        elif params["type"].lower() == "raw":
            dataset_params = {
                "path": params["path"],
                "dtype": params["dtype"],
                "detector_size_raw": params["detectorSizeRaw"],
                "crop_detector_to": params["cropDetectorTo"],
                "tileshape": params["tileshape"],
                "scan_size": params["scanSize"],
            }
        elif params["type"].lower() == "mib":
            dataset_params = {
                "path": params["path"],
                "tileshape": params["tileshape"],
                "scan_size": params["scanSize"],
            }
        elif params["type"].lower() == "blo":
            dataset_params = {
                "path": params["path"],
                "tileshape": params["tileshape"],
            }
        elif params["type"].lower() == "k2is":
            dataset_params = {
                "path": params["path"],
                "scan_size": params["scanSize"],
            }
        try:
            ds = await run_blocking(dataset.load, filetype=params["type"], **dataset_params)
            await run_blocking(ds.check_valid)
        except DataSetException as e:
            msg = Message(self.data).create_dataset_error(uuid, str(e))
            log_message(msg)
            self.write(msg)
            return
        self.data.register_dataset(
            uuid=uuid,
            dataset=ds,
            params=request_data['dataset'],
        )
        msg = Message(self.data).create_dataset(dataset=uuid)
        log_message(msg)
        self.write(msg)
        self.event_registry.broadcast_event(msg)


class DataSetPreviewHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    async def get_preview_image(self, dataset_uuid):
        ds = self.data.get_dataset(dataset_uuid)
        job = SumFramesJob(dataset=ds)

        executor = self.data.get_executor()

        log.info("creating preview for dataset %s" % dataset_uuid)

        futures = []
        for task in job.get_tasks():
            submit_kwargs = {}
            futures.append(
                executor.client.submit(task, **submit_kwargs)
            )
        log.info("preview futures created")

        full_result = np.zeros(shape=ds.shape[2:], dtype="float32")
        async for future, result in dd.as_completed(futures, with_results=True):
            for tile in result:
                tile.copy_to_result(full_result)
        log.info("preview done, encoding image (dtype=%s)", full_result.dtype)
        visualized = await run_blocking(
            visualize_simple,
            full_result,
            colormap=cm.gist_earth,
        )
        image = await run_blocking(
            encode_image,
            visualized
        )
        log.info("image encoded, sending response")
        return image.read()

    def set_max_expires(self):
        cache_time = 86400 * 365 * 10
        self.set_header("Expires", datetime.datetime.utcnow() +
                        datetime.timedelta(seconds=cache_time))
        self.set_header("Cache-Control", "max-age=" + str(cache_time))

    async def get(self, uuid):
        """
        make a preview and return it as HTTP response
        """
        self.set_header('Content-Type', 'image/png')
        self.set_max_expires()
        image = await self.get_preview_image(uuid)
        self.write(image)


class DataSetPickHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    async def pick_frame(self, dataset_uuid, x, y):
        ds = self.data.get_dataset(dataset_uuid)
        x = int(x)
        y = int(y)
        slice_ = Slice(
            origin=(y, x, 0, 0),
            shape=(1, 1, ds.shape[2], ds.shape[3])
        )
        job = PickFrameJob(dataset=ds, slice_=slice_)

        executor = self.data.get_executor()

        log.info("picking %d/%d from %s", x, y, dataset_uuid)

        futures = []
        for task in job.get_tasks():
            submit_kwargs = {}
            futures.append(
                executor.client.submit(task, **submit_kwargs)
            )

        full_result = np.zeros(shape=ds.shape[2:], dtype=ds.dtype)
        async for future, result in dd.as_completed(futures, with_results=True):
            for tile in result:
                tile.copy_to_result(full_result)
        log.info("picking %d/%d done, encoding image (dtype=%s)", x, y, full_result.dtype)
        visualized = await run_blocking(
            visualize_simple,
            full_result,
            colormap=cm.gist_earth,
        )
        image = await run_blocking(
            encode_image,
            visualized
        )
        log.info("image %d/%d encoded, sending response", x, y)
        return image.read()

    def set_max_expires(self):
        cache_time = 86400 * 365 * 10
        self.set_header("Expires", datetime.datetime.utcnow() +
                        datetime.timedelta(seconds=cache_time))
        self.set_header("Cache-Control", "max-age=" + str(cache_time))

    async def get(self, uuid, x, y):
        """
        pick a raw frame and return it as HTTP response
        """
        x = int(x)
        y = int(y)
        self.set_header('Content-Type', 'image/png')
        self.set_max_expires()
        image = await self.pick_frame(uuid, x, y)
        self.write(image)


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
            "localCores": self.get_local_cores(),
            "cwd": os.getcwd(),
            "separator": os.sep,
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

    def set_executor(self, executor, params):
        if self.executor is not None:
            self.executor.close()
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

    def verify_datasets(self):
        for uuid, params in self.datasets.items():
            dataset = params["dataset"]
            try:
                dataset.check_valid()
            except DataSetException:
                self.remove_dataset(uuid)

    def remove_dataset(self, uuid):
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
            del self.jobs[job_id]
        for job in jobs_to_remove:
            del self.job_to_id[job]

    def register_job(self, uuid, job):
        assert uuid not in self.jobs
        self.jobs[uuid] = job
        self.job_to_id[job] = uuid
        return self

    def get_job(self, uuid):
        return self.jobs[uuid]

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

    def serialize_dataset(self, dataset_id):
        dataset = self.datasets[dataset_id]
        return {
            "id": dataset_id,
            "params": {
                **dataset["params"]["params"],
                "shape": dataset["dataset"].shape,
            }
        }

    def serialize_datasets(self):
        return [
            self.serialize_dataset(dataset_id)
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
            dask_client = await AioClient(address=connection['address'])
            executor = DaskJobExecutor(client=dask_client, is_local=connection['isLocal'])
        elif connection["type"].lower() == "local":
            # NOTE: we can't use DaskJobExecutor.make_local as it doesn't use AioClient
            # which then conflicts with LocalCluster(asynchronous=True)
            # error message: "RuntimeError: Non-thread-safe operation invoked on an event loop
            # other than the current one"
            # related: debugging via env var PYTHONASYNCIODEBUG=1
            cluster_kwargs = {
                "threads_per_worker": 1,
                "asynchronous": True,
            }
            if "numWorkers" in connection:
                cluster_kwargs.update({"n_workers": connection["numWorkers"]})
            cluster = dd.LocalCluster(**cluster_kwargs)
            dask_client = await AioClient(address=cluster)
            executor = DaskJobExecutor(client=dask_client, is_local=True)
        self.data.set_executor(executor, request_data)
        msg = Message(self.data).initial_state(
            jobs=self.data.serialize_jobs(),
            datasets=self.data.serialize_datasets(),
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

    def get(self):
        path = self.request.arguments['path']
        assert len(path) == 1
        path = path[0].decode("utf8")
        assert os.path.isdir(path)
        path = os.path.abspath(path)
        names = os.listdir(path)
        dirs = []
        files = []
        names = [".."] + names
        for name in names:
            full_path = os.path.join(path, name)
            try:
                s = os.stat(full_path)
            except FileNotFoundError:
                # this can happen either because of a TOCTOU-like race condition
                # or for example for things like broken softlinks
                continue
            res = {"name": name, "stat": s}
            if stat.S_ISDIR(s.st_mode):
                dirs.append(res)
            else:
                files.append(res)
        msg = Message(self.data).directory_listing(path, files=files, dirs=dirs)
        self.write(msg)


# shared state:
event_registry = EventRegistry()
data = SharedData()


def make_app():
    settings = {
        "static_path": os.path.join(os.path.dirname(__file__), "client"),
    }
    return tornado.web.Application([
        (r"/", IndexHandler, {"data": data, "event_registry": event_registry}),
        (r"/api/datasets/([^/]+)/", DataSetDetailHandler, {
            "data": data,
            "event_registry": event_registry
        }),
        (r"/api/datasets/([^/]+)/preview/", DataSetPreviewHandler, {
            "data": data,
            "event_registry": event_registry
        }),
        (r"/api/datasets/([^/]+)/pick/([0-9]+)/([0-9]+)/", DataSetPickHandler, {
            "data": data,
            "event_registry": event_registry
        }),
        (r"/api/browse/localfs/", LocalFSBrowseHandler, {
            "data": data,
            "event_registry": event_registry
        }),
        (r"/api/jobs/([^/]+)/", JobDetailHandler, {
            "data": data,
            "event_registry": event_registry
        }),
        (r"/api/events/", ResultEventHandler, {"data": data, "event_registry": event_registry}),
        (r"/api/config/", ConfigHandler, {
            "data": data,
            "event_registry": event_registry
        }),
        (r"/api/config/connection/", ConnectHandler, {
            "data": data,
            "event_registry": event_registry,
        }),
    ], **settings)


def sig_exit(signum, frame):
    tornado.ioloop.IOLoop.instance().add_callback_from_signal(do_stop)


def do_stop():
    log.warning("Exiting...")
    try:
        data.get_executor().close()
        tornado.ioloop.IOLoop.instance().stop()
    finally:
        sys.exit(0)


def main(host, port):
    logging.basicConfig(
        level=logging.DEBUG,
        format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s",
    )
    log.info("listening on %s:%s" % (host, port))
    app = make_app()
    app.listen(address=host, port=port)


def run(host, port):
    main(host, port)
    loop = asyncio.get_event_loop()
    signal.signal(signal.SIGINT, sig_exit)
    loop.run_forever()


if __name__ == "__main__":
    main("0.0.0.0", 9000)
