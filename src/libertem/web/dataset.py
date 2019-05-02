import logging

import tornado.web

from libertem.io.dataset import load, detect
from .base import CORSMixin, log_message
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
        # TODO: validate request_data
        # let's start simple:
        ds_type = params["type"].lower()
        assert ds_type in ["hdfs", "hdf5", "raw", "mib", "blo", "k2is", "ser",
                           "frms6", "empad"]
        if ds_type == "hdfs":
            dataset_params = {
                "index_path": params["path"],
                "tileshape": params["tileshape"],
                "host": "localhost",  # FIXME: config param
                "port": 8020,  # FIXME: config param
            }
        elif ds_type == "hdf5":
            dataset_params = {
                "path": params["path"],
                "ds_path": params["ds_path"],
                "tileshape": params["tileshape"],
            }
        elif ds_type == "raw":
            dataset_params = {
                "path": params["path"],
                "dtype": params["dtype"],
                "detector_size": params["detector_size"],
                "enable_direct": params["enable_direct"],
                "tileshape": params["tileshape"],
                "scan_size": params["scan_size"],
            }
        elif ds_type == "empad":
            dataset_params = {
                "path": params["path"],
                "scan_size": params["scan_size"],
            }
        elif ds_type == "mib":
            dataset_params = {
                "path": params["path"],
                "tileshape": params["tileshape"],
                "scan_size": params["scan_size"],
            }
        elif ds_type == "blo":
            dataset_params = {
                "path": params["path"],
                "tileshape": params["tileshape"],
            }
        elif ds_type == "k2is":
            dataset_params = {
                "path": params["path"],
            }
        elif ds_type == "ser":
            dataset_params = {
                "path": params["path"],
            }
        elif ds_type == "frms6":
            dataset_params = {
                "path": params["path"],
            }
        try:
            executor = self.data.get_executor()
            ds = await executor.run_function(load, filetype=params["type"], **dataset_params)
            ds = await executor.run_function(ds.initialize)
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

        params = await executor.run_function(detect, path=path)
        if not params:
            msg = Message(self.data).dataset_detect_failed(path=path)
            log_message(msg)
            self.write(msg)
            return
        params['type'] = params['type'].upper()
        msg = Message(self.data).dataset_detect(params=params)
        log_message(msg)
        self.write(msg)
