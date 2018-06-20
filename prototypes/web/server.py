import copy
import os
from io import BytesIO
from libertem.dataset.hdfs import BinaryHDFSDataSet
from libertem.executor.dask import DaskJobExecutor
from libertem.job.masks import ApplyMasksJob
from libertem.masks import ring
import numpy as np
import tornado.web
import tornado.websocket
import tornado.ioloop
import tornado.escape
from dask import distributed as dd
from distributed.asyncio import AioClient
from PIL import Image
from matplotlib import cm


class Message(object):
    """
    possible messages - the translation of our python datatypes to json types
    """

    @classmethod
    def initial_state(cls, jobs, datasets):
        # TODO: what else is part of the initial state?
        return {
            "messageType": "INITIAL_STATE",
            "datasets": datasets,
            "jobs": jobs,
        }

    @classmethod
    def start_job(cls, job_id):
        # TODO: job parameters?
        return {
            "messageType": "START_JOB",
            "job": job_id,
        }

    @classmethod
    def finish_job(cls, job_id, num_images):
        return {
            "messageType": "FINISH_JOB",
            "job": job_id,
            "followup": {
                "numMessages": num_images,
            },
        }

    @classmethod
    def task_result(cls, job_id, num_images):
        return {
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
        self.write_message(Message.initial_state(
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
                return ring(
                    centerX=cx, centerY=cy,
                    imageSizeX=frame_size[1],
                    imageSizeY=frame_size[0],
                    radius=ro,
                    radius_inner=ri
                )
            return _inner

        factories = []
        for params in mask_params:
            kwargs = copy.deepcopy(params)
            shape = kwargs.pop('shape')
            assert shape == 'ring'
            for p in ['cx', 'cy', 'ri', 'ro']:
                assert p in kwargs
            assert len(kwargs.keys()) == 4
            factories.append(_make_ring(**kwargs))
        return factories

    async def result_images(self, full_result, save_kwargs=None):
        colormap = cm.gist_earth
        # colormap = cm.viridis
        save_kwargs = save_kwargs or {}

        def _encode_image(result):
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

        # TODO: do encoding work in a thread pool
        return [_encode_image(full_result[idx]) for idx in range(full_result.shape[0])]

    async def put(self, uuid):
        request_data = tornado.escape.json_decode(self.request.body)
        params = request_data['job']
        ds = self.data.get_dataset(params['dataset'])
        mask_factories = self.make_mask_factories(params['masks'], frame_size=ds.shape[2:])
        job = ApplyMasksJob(dataset=ds, mask_factories=mask_factories)
        self.data.register_job(uuid=uuid, job=job)

        dask_client = await AioClient("tcp://localhost:8786")
        executor = DaskJobExecutor(client=dask_client, is_local=True)

        futures = []
        for task in job.get_tasks():
            submit_kwargs = {}
            futures.append(
                executor.client.submit(task, **submit_kwargs)
            )
        self.write({"status": "ok"})
        self.finish()
        self.event_registry.broadcast_event(Message.start_job(
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
            self.event_registry.broadcast_event(Message.task_result(
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
        self.event_registry.broadcast_event(Message.finish_job(
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
        params = request_data['dataset']
        # TODO: validate request_data
        # let's start simple:
        assert params['type'] == "HDFS"
        ds = BinaryHDFSDataSet(
            index_path=params['path'],  # TODO: validate! kerberosify! etc.
            host='localhost',
            port=8020,
            tileshape=params['tileshape'],
        )
        self.data.register_dataset(
            uuid=uuid,
            dataset=ds,
        )
        self.write({
            "status": "ok",
            "dataset": uuid,
        })


class SharedData(object):
    def __init__(self):
        self.datasets = {}
        self.jobs = {}
        self.job_to_id = {}
        self.dataset_to_id = {}

    def register_dataset(self, uuid, dataset):
        assert uuid not in self.datasets
        self.datasets[uuid] = dataset
        self.dataset_to_id[dataset] = uuid
        return self

    def get_dataset(self, uuid):
        return self.datasets[uuid]

    def register_job(self, uuid, job):
        assert uuid not in self.jobs
        self.jobs[uuid] = job
        self.job_to_id[job] = uuid
        return self

    def get_job(self, uuid):
        return self.jobs[uuid]

    def serialize_jobs(self):
        return [
            {
                "job": job_id,
                "dataset": self.dataset_to_id[job.dataset],
            }
            for job_id, job in self.jobs.items()
        ]

    def serialize_datasets(self):
        return [
            {
                # TODO: fill values
                "dataset": dataset_id,
                "name": "",
                "path": "",
                "tileshape": [],
                "type": "",
            }
            for dataset_id, dataset in self.datasets.items()
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
        "static_path": os.path.join(os.path.dirname(__file__), "static"),
    }
    return tornado.web.Application([
        (r"/", IndexHandler, {"data": data, "event_registry": event_registry}),
        (r"/datasets/([^/]+)/", DataSetDetailHandler, {
            "data": data,
            "event_registry": event_registry
        }),
        (r"/jobs/([^/]+)/", JobDetailHandler, {
            "data": data,
            "event_registry": event_registry
        }),
        (r"/events/", ResultEventHandler, {"data": data, "event_registry": event_registry}),
    ], **settings)


if __name__ == "__main__":
    app = make_app()
    app.listen(9000)
    tornado.ioloop.IOLoop.current().start()
