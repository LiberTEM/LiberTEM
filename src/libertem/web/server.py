import datetime
import copy
from io import BytesIO

import numpy as np
from matplotlib import cm
from dask import distributed as dd
from distributed.asyncio import AioClient
import tornado.web
import tornado.websocket
import tornado.ioloop
import tornado.escape
from PIL import Image

from libertem.executor.dask import DaskJobExecutor
from libertem.job.masks import ApplyMasksJob
from libertem.job.sum import SumFramesJob
from libertem import masks, dataset


def _encode_image(result, colormap, save_kwargs):
    # TODO: only normalize across the area where we already have values
    # can be accomplished by calculating min/max over are that was
    # affected by the result tiles. for now, ignoring 0 works fine
    max_ = np.max(result)
    min_ = np.min(result[result > 0])

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

    def create_dataset(self, dataset):
        return {
            "status": "ok",
            "messageType": "CREATE_DATASET",
            "dataset": dataset,
            "details": self.data.serialize_dataset(dataset),
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


class ResultEventHandler(tornado.websocket.WebSocketHandler):
    def initialize(self, data, event_registry):
        self.registry = event_registry
        self.data = data

    def check_origin(self, origin):
        return True  # TODO XXX FIXME !!!

    def open(self):
        self.registry.add_handler(self)
        self.write_message(Message(self.data).initial_state(
            jobs=self.data.serialize_jobs(),
            datasets=self.data.serialize_datasets(),
        ))

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
    def set_default_headers(self):
        self.set_header("Access-Control-Allow-Origin", "*")  # XXX FIXME TODO!!!
        # self.set_header("Access-Control-Allow-Headers", "x-requested-with")
        self.set_header('Access-Control-Allow-Methods', 'PUT, POST, GET, OPTIONS')

    def options(self, *args):
        """
        for CORS pre-flight requests, no body returned
        """
        self.set_status(204)
        self.finish()


class JobDetailHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    def make_mask_factories(self, mask_params, frame_size):
        def _make_ring(cx, cy, ri, ro):
            def _inner():
                return masks.ring(
                    centerX=cx, centerY=cy,
                    imageSizeX=frame_size[1],
                    imageSizeY=frame_size[0],
                    radius=ro,
                    radius_inner=ri
                )
            return _inner

        def _make_disk(cx, cy, r):
            def _inner():
                return masks.circular(
                    centerX=cx, centerY=cy,
                    imageSizeX=frame_size[1],
                    imageSizeY=frame_size[0],
                    radius=r,
                )
            return _inner

        fn_by_shape = {
            "disk": _make_disk,
            "ring": _make_ring,
        }

        factories = []
        for params in mask_params:
            kwargs = copy.deepcopy(params)
            shape = kwargs.pop('shape')
            assert shape in fn_by_shape
            if shape == "ring":
                assert len(kwargs.keys()) == 4
                for p in ['cx', 'cy', 'ri', 'ro']:
                    assert p in kwargs
            elif shape == "disk":
                assert len(kwargs.keys()) == 3
                for p in ['cx', 'cy', 'r']:
                    assert p in kwargs
            factories.append(fn_by_shape[shape](**kwargs))
        return factories

    async def result_images(self, full_result, save_kwargs=None):
        colormap = cm.gist_earth
        # colormap = cm.viridis
        save_kwargs = save_kwargs or {}

        # TODO: do encoding work in a thread pool
        return [_encode_image(full_result[idx], colormap, save_kwargs)
                for idx in range(full_result.shape[0])]

    async def put(self, uuid):
        request_data = tornado.escape.json_decode(self.request.body)
        params = request_data['job']
        ds = self.data.get_dataset(params['dataset'])
        mask_factories = self.make_mask_factories(params['masks'], frame_size=ds.shape[2:])
        job = ApplyMasksJob(dataset=ds, mask_factories=mask_factories)
        self.data.register_job(uuid=uuid, job=job)

        # TODO: config param
        dask_client = await AioClient("tcp://localhost:8786")
        executor = DaskJobExecutor(client=dask_client, is_local=True)

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
        self.event_registry.broadcast_event(Message(self.data).start_job(
            job_id=uuid,
        ))

        full_result = np.zeros(shape=(len(mask_factories),) + tuple(ds.shape[:2]))
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
            images = await self.result_images(
                full_result,
                save_kwargs={'format': 'jpeg', 'quality': 65},
            )

            # NOTE: make sure the following broadcast_event messages are sent atomically!
            # (that is: keep the code below synchronous, and only send the messages
            # once the images have finished encoding, and then send all at once)
            self.event_registry.broadcast_event(Message(self.data).task_result(
                job_id=uuid,
                num_images=len(images),
            ))
            for image in images:
                raw_bytes = image.read()
                self.event_registry.broadcast_event(raw_bytes, binary=True)
        images = await self.result_images(
            full_result,
            save_kwargs={'format': 'png'},
        )
        self.event_registry.broadcast_event(Message(self.data).finish_job(
            job_id=uuid,
            num_images=len(images),
        ))
        for image in images:
            raw_bytes = image.read()
            self.event_registry.broadcast_event(raw_bytes, binary=True)


class DataSetDetailHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    def put(self, uuid):
        request_data = tornado.escape.json_decode(self.request.body)
        params = request_data['dataset']['params']
        # TODO: validate request_data
        # let's start simple:
        assert params['type'].lower() in ["hdfs", "hdf5"]
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
        ds = dataset.load(filetype=params["type"], **dataset_params)
        self.data.register_dataset(
            uuid=uuid,
            dataset=ds,
            params=request_data['dataset'],
        )
        self.write(Message(self.data).create_dataset(dataset=uuid))
        self.event_registry.broadcast_event(Message(self.data).create_dataset(dataset=uuid))


class DataSetPreviewHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    async def get_preview_image(self, dataset_uuid):
        ds = self.data.get_dataset(dataset_uuid)
        job = SumFramesJob(dataset=ds)

        # TODO: config param
        # TODO: share dask client with JobDetailHandler
        dask_client = await AioClient("tcp://localhost:8786")

        # TODO: is_local -> config param?
        executor = DaskJobExecutor(client=dask_client, is_local=True)

        futures = []
        for task in job.get_tasks():
            submit_kwargs = {}
            futures.append(
                executor.client.submit(task, **submit_kwargs)
            )

        full_result = np.zeros(shape=ds.shape[2:])
        async for future, result in dd.as_completed(futures, with_results=True):
            for tile in result:
                tile.copy_to_result(full_result)
        image = _encode_image(
            full_result,
            colormap=cm.gist_earth,
            save_kwargs={'format': 'png'},
        )
        return image.read()

    def set_max_expires(self):
        cache_time = 86400 * 365 * 10
        self.set_header("Expires", datetime.datetime.utcnow() +
                        datetime.timedelta(seconds=cache_time))
        self.set_header("Cache-Control", "max-age=" + str(cache_time))

    async def get(self, uuid):
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
        print(dataset)
        return {
            "id": dataset_id,
            "name": dataset["params"]["name"],
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


class IndexHandler(tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    def get(self):
        self.render("templates/index.html")


# shared state:
event_registry = EventRegistry()
data = SharedData()


def make_app():
    settings = {
        # "static_path": os.path.join(os.path.dirname(__file__), "static"),
    }
    return tornado.web.Application([
        # (r"/", IndexHandler, {"data": data, "event_registry": event_registry}),
        (r"/datasets/([^/]+)/", DataSetDetailHandler, {
            "data": data,
            "event_registry": event_registry
        }),
        (r"/datasets/([^/]+)/preview/", DataSetPreviewHandler, {
            "data": data,
            "event_registry": event_registry
        }),
        (r"/jobs/([^/]+)/", JobDetailHandler, {
            "data": data,
            "event_registry": event_registry
        }),
        (r"/events/", ResultEventHandler, {"data": data, "event_registry": event_registry}),
    ], **settings)


def run(port):
    app = make_app()
    app.listen(port)
    tornado.ioloop.IOLoop.current().start()


if __name__ == "__main__":
    run(9000)
