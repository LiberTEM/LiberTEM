import time
import logging
import functools

import tornado.web

from libertem.io.dataset import load, detect, get_dataset_cls
from libertem.common.numba import prime_numba_cache
from .base import CORSMixin, log_message
from libertem.utils.async_utils import sync_to_async
from .messages import Message
from .state import SharedState

log = logging.getLogger(__name__)


class DataSetDetailHandler(CORSMixin, tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.dataset_state = state.dataset_state
        self.event_registry = event_registry

    async def delete(self, uuid):
        try:
            self.dataset_state[uuid]
        except KeyError:
            self.set_status(404, "dataset with uuid %s not found" % uuid)
            return
        await self.dataset_state.remove(uuid)
        msg = Message(self.state).delete_dataset(uuid)
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)

    async def prime_numba_caches(self, ds):
        executor = self.state.executor_state.get_executor()

        log.info("starting warmup")

        t0 = time.time()
        # first: make sure the jited functions used for I/O are compiled
        # by running a single-core workload on each host:
        await executor.run_each_host(functools.partial(prime_numba_cache, ds=ds))

        t1 = time.time()

        # second: make sure each worker *process* has the jited functions
        # loaded from the cache
        # XXX doesn't seem to be needed actually!
        # await executor.run_each_worker(functools.partial(prime_numba_cache, ds=ds))

        # t2 = time.time()

        log.info("warmup done, took %.3fs", (t1 - t0))

    async def put(self, uuid):
        request_data = tornado.escape.json_decode(self.request.body)
        params = request_data['dataset']['params']
        params["type"] = ds_type = params["type"].upper()
        cls = get_dataset_cls(ds_type)
        ConverterCls = cls.get_msg_converter()
        converter = ConverterCls()
        try:
            dataset_params = converter.to_python(params)
            executor = self.state.executor_state.get_executor()

            ds = await load(filetype=cls, executor=executor, enable_async=True, **dataset_params)

            await self.prime_numba_caches(ds)

            self.dataset_state.register(
                uuid=uuid,
                dataset=ds,
                params=request_data['dataset'],
                converted=dataset_params,
            )
            details = await self.dataset_state.serialize(dataset_id=uuid)
            msg = Message(self.state).create_dataset(dataset=uuid, details=details)
            log_message(msg)
            self.write(msg)
            self.event_registry.broadcast_event(msg)
        except Exception as e:
            if uuid in self.dataset_state:
                await self.dataset_state.remove(uuid)
            msg = Message(self.state).create_dataset_error(uuid, str(e))
            log_message(msg, exception=True)
            self.write(msg)
            return


class DataSetOpenSchema(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    def get(self):
        try:
            ds_type = self.request.arguments['type'][0].decode("utf8")
            cls = get_dataset_cls(ds_type)
            ConverterCls = cls.get_msg_converter()
            converter = ConverterCls()
            schema = converter.SCHEMA
            msg = Message(self.state).dataset_schema(ds_type, schema)
            log_message(msg)
            self.write(msg)
        except Exception as e:
            msg = Message(self.state).dataset_schema_failed(ds_type, str(e))
            log_message(msg, exception=True)
            self.write(msg)
            return


class DataSetDetectHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def get(self):
        path = self.request.arguments['path'][0].decode("utf8")
        executor = self.state.executor_state.get_executor()

        detected_params = await sync_to_async(
            detect, path=path, executor=executor.ensure_sync()
        )

        if not detected_params:
            msg = Message(self.state).dataset_detect_failed(path=path)
            log_message(msg)
            self.write(msg)
            return
        params = detected_params["parameters"]
        info = {}
        if "info" in detected_params:
            info = detected_params["info"]
        params.update({"type": detected_params["type"].upper()})
        info.update({"type": detected_params["type"].upper()})
        msg = Message(self.state).dataset_detect(params=params, info=info)
        log_message(msg)
        self.write(msg)
