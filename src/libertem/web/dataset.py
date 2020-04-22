import logging

import tornado.web

from libertem.io.dataset import load, detect, get_dataset_cls
from .base import CORSMixin, log_message
from libertem.utils.async_utils import run_blocking
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

            await executor.run_function(ds.check_valid)
            self.dataset_state.register(
                uuid=uuid,
                dataset=ds,
                params=request_data['dataset'],
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

        detected_params = await run_blocking(
            detect, path=path, executor=executor.ensure_sync()
        )

        if not detected_params:
            msg = Message(self.state).dataset_detect_failed(path=path)
            log_message(msg)
            self.write(msg)
            return
        params = detected_params["parameters"]
        params.update({"type": detected_params["type"].upper()})
        if "info" in detected_params:
            params.update(detected_params["info"])
        msg = Message(self.state).dataset_detect(params=params)
        log_message(msg)
        self.write(msg)
