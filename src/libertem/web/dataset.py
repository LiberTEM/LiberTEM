import logging

import tornado.web

from libertem.io.dataset import load, detect, get_dataset_cls
from .base import CORSMixin, log_message, run_blocking
from .messages import Message

log = logging.getLogger(__name__)


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
        params["type"] = ds_type = params["type"].upper()
        cls = get_dataset_cls(ds_type)
        ConverterCls = cls.get_msg_converter()
        converter = ConverterCls()
        try:
            dataset_params = converter.to_python(params)
            executor = self.data.get_executor()
            ds = await executor.run_function(load, filetype=cls, **dataset_params)
            ds = await run_blocking(ds.initialize, executor=executor.ensure_sync())
            available_workers = await executor.get_available_workers()
            ds.set_num_cores(len(available_workers))
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
            if self.data.has_dataset(uuid):
                await self.data.remove_dataset(uuid)
            msg = Message(self.data).create_dataset_error(uuid, str(e))
            log_message(msg, exception=True)
            self.write(msg)
            return


class DataSetOpenSchema(tornado.web.RequestHandler):
    def initialize(self, data, event_registry):
        self.data = data
        self.event_registry = event_registry

    def get(self):
        try:
            ds_type = self.request.arguments['type'][0].decode("utf8")
            cls = get_dataset_cls(ds_type)
            ConverterCls = cls.get_msg_converter()
            converter = ConverterCls()
            schema = converter.SCHEMA
            msg = Message(self.data).dataset_schema(ds_type, schema)
            log_message(msg)
            self.write(msg)
        except Exception as e:
            msg = Message(self.data).dataset_schema_failed(ds_type, str(e))
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

        params = await run_blocking(detect, path=path, executor=executor.ensure_sync())

        if not params:
            msg = Message(self.data).dataset_detect_failed(path=path)
            log_message(msg)
            self.write(msg)
            return
        params['type'] = params['type'].upper()
        msg = Message(self.data).dataset_detect(params=params)
        log_message(msg)
        self.write(msg)
