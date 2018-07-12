import os
import sys
import datetime
import logging
import asyncio
import signal
import psutil
from functools import partial
from io import BytesIO

import numpy as np
from matplotlib import cm
from dask import distributed as dd
from distributed.asyncio import AioClient
import tornado.web
import tornado.gen
import tornado.websocket
import tornado.ioloop
import tornado.escape
from PIL import Image

import libertem
from libertem.executor.dask import DaskJobExecutor
from libertem.io.dataset.base import DataSetException
from libertem.io import dataset
from libertem.job.masks import ApplyMasksJob
from libertem.job.sum import SumFramesJob
from libertem import masks


log = logging.getLogger(__name__)


def log_message(message):
    if "job" in message:
        log.info("message: %s (job=%s)" % (message["messageType"], message["job"]))
    else:
        log.info("message: %s" % message["messageType"])


def _encode_image(result, colormap, save_kwargs):
    # TODO: only normalize across the area where we already have values
    # can be accomplished by calculating min/max over are that was
    # affected by the result tiles. for now, ignoring 0 works fine
    result = result.astype(np.float32)
    max_ = np.max(result)
    result_gt_zero = result[result > 0]
    if len(result_gt_zero) == 0:
        min_ = 0
    else:
        min_ = np.min(result_gt_zero)

    normalized = result - min_
    if max_ > 0:
        normalized = normalized / max_
    # see also: https://stackoverflow.com/a/10967471/540644
    im = Image.fromarray(colormap(normalized, bytes=True))
    buf = BytesIO()
    im = im.convert(mode="RGB")
    im.save(buf, **save_kwargs)
    buf.seek(0)
    return buf


async def result_images(full_result, save_kwargs=None):
    colormap = cm.gist_earth
    # colormap = cm.viridis
    save_kwargs = save_kwargs or {}

    futures = [
        run_blocking(_encode_image, full_result[idx], colormap, save_kwargs)
        for idx in range(full_result.shape[0])
    ]

    results = await asyncio.gather(*futures)
    return results


async def run_blocking(fn, *args, **kwargs):
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

    def finish_job(self, job_id, num_images):
        return {
            "status": "ok",
            "messageType": "FINISH_JOB",
            "job": job_id,
            "details": self.data.serialize_job(job_id),
            "followup": {
                "numMessages": num_images,
            },
        }

    def task_result(self, job_id, num_images):
        return {
            "status": "ok",
            "messageType": "TASK_RESULT",
            "job": job_id,
            "followup": {
                "numMessages": num_images,
            },
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
            images = yield full_result

            # NOTE: make sure the following broadcast_event messages are sent atomically!
            # (that is: keep the code below synchronous, and only send the messages
            # once the images have finished encoding, and then send all at once)
            msg = Message(self.data).task_result(
                job_id=uuid,
                num_images=len(images),
            )
            log_message(msg)
            self.event_registry.broadcast_event(msg)
            for image in images:
                raw_bytes = image.read()
                self.event_registry.broadcast_event(raw_bytes, binary=True)
        images = yield full_result
        msg = Message(self.data).finish_job(
            job_id=uuid,
            num_images=len(images),
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

    def make_mask_factories(self, analysis, dtype, frame_size):
        def _make_ring(cx, cy, ri, ro, shape):
            def _ring_inner():
                return masks.ring(
                    centerX=cx, centerY=cy,
                    imageSizeX=frame_size[1],
                    imageSizeY=frame_size[0],
                    radius=ro,
                    radius_inner=ri
                )
            return [_ring_inner]

        def _make_disk(cx, cy, r, shape):
            def _disk_inner():
                return masks.circular(
                    centerX=cx, centerY=cy,
                    imageSizeX=frame_size[1],
                    imageSizeY=frame_size[0],
                    radius=r,
                )
            return [_disk_inner]

        def _make_point(cx, cy, shape):
            def _point_inner():
                a = np.zeros(frame_size)
                a[int(cy), int(cx)] = 1
                return a.astype(dtype)
            return [_point_inner]

        def _make_com(cx, cy, r, shape):
            disk_mask = masks.circular(
                centerX=cx, centerY=cy,
                imageSizeX=frame_size[1],
                imageSizeY=frame_size[0],
                radius=r,
            )
            return [
                lambda: disk_mask,
                lambda: masks.gradient_x(
                    imageSizeX=frame_size[1],
                    imageSizeY=frame_size[0],
                    dtype=dtype,
                ) * (np.ones(frame_size) * disk_mask),
                lambda: masks.gradient_y(
                    imageSizeX=frame_size[1],
                    imageSizeY=frame_size[0],
                    dtype=dtype,
                ) * (np.ones(frame_size) * disk_mask),
            ]

        fn_by_type = {
            "APPLY_DISK_MASK": _make_disk,
            "APPLY_RING_MASK": _make_ring,
            "APPLY_POINT_SELECTOR": _make_point,
            "CENTER_OF_MASS": _make_com,
        }
        return fn_by_type[analysis['type']](**analysis['parameters'])

    async def put(self, uuid):
        request_data = tornado.escape.json_decode(self.request.body)
        params = request_data['job']
        ds = self.data.get_dataset(params['dataset'])
        analysis = params['analysis']
        if analysis['type'] == "SUM_FRAMES":
            return await self.start_sum_frames_job(
                uuid=uuid,
                params=params,
                analysis=analysis,
                ds=ds,
            )
        else:
            return await self.start_mask_job(
                uuid=uuid,
                params=params,
                analysis=analysis,
                ds=ds,
            )

    async def start_sum_frames_job(self, uuid, params, analysis, ds):
        job = SumFramesJob(dataset=ds)
        full_result = np.zeros(shape=tuple(ds.shape[:2]))
        job_runner = self.run_job(
            full_result=full_result,
            uuid=uuid, ds=ds, job=job,
        )
        try:
            await job_runner.asend(None)
            while True:
                images = await self.visualize(
                    full_result,
                    analysis,
                    save_kwargs={'format': 'png'},
                )
                await job_runner.asend(images)
        except StopAsyncIteration:
            pass

    async def start_mask_job(self, uuid, params, analysis, ds):
        dtype = np.dtype(ds.dtype).kind == 'f' and ds.dtype or "float32"
        mask_factories = self.make_mask_factories(analysis, dtype, frame_size=ds.shape[2:])
        job = ApplyMasksJob(dataset=ds, mask_factories=mask_factories)
        full_result = np.zeros(shape=(len(mask_factories),) + tuple(ds.shape[:2]))
        job_runner = self.run_job(
            full_result=full_result,
            uuid=uuid, ds=ds, job=job,
        )
        try:
            await job_runner.asend(None)
            while True:
                images = await self.visualize(
                    full_result,
                    analysis,
                    save_kwargs={'format': 'png'},
                )
                await job_runner.asend(images)
        except StopAsyncIteration:
            pass

    async def visualize_com(self, full_result, analysis, save_kwargs=None):
        img_sum, img_x, img_y = full_result[0], full_result[1], full_result[2]
        ref_x = analysis["parameters"]["cx"]
        ref_y = analysis["parameters"]["cy"]
        x_centers = np.divide(img_x, img_sum) - ref_x
        y_centers = np.divide(img_y, img_sum) - ref_y
        centers = np.stack((x_centers, y_centers), axis=0)
        log.debug("full_result.shape: %r", full_result.shape)
        log.debug("centers.shape: %r", centers.shape)
        return await result_images(centers, save_kwargs)

    async def visualize(self, full_result, analysis, save_kwargs=None):
        if analysis['type'] in {'SUM_FRAMES', 'APPLY_DISK_MASK',
                                'APPLY_RING_MASK', 'APPLY_POINT_SELECTOR'}:
            return await result_images(full_result, save_kwargs)
        elif analysis['type'] in {'CENTER_OF_MASS'}:
            return await self.visualize_com(full_result, analysis, save_kwargs)

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
        assert params['type'].lower() in ["hdfs", "hdf5", "raw", "mib"]
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
                "files_pattern": params["filesPattern"],
                "tileshape": params["tileshape"],
                "scan_size": params["scanSize"],
            }
        try:
            ds = dataset.load(filetype=params["type"], **dataset_params)
            ds.check_valid()
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

        full_result = np.zeros(shape=ds.shape[2:])
        async for future, result in dd.as_completed(futures, with_results=True):
            for tile in result:
                tile.copy_to_result(full_result)
        log.info("preview done, encoding image (dtype=%s)", full_result.dtype)
        image = await run_blocking(
            _encode_image,
            full_result,
            colormap=cm.gist_earth,
            save_kwargs={'format': 'png'},
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
