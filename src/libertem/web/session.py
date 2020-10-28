import logging
import uuid
from functools import partial

import tornado.web
from libertem.executor.base import AsyncAdapter, sync_to_async
from libertem.executor.dask import DaskJobExecutor, cluster_spec
from libertem.io.dataset import detect, get_dataset_cls, load
from libertem.utils.async_utils import run_blocking
from libertem.utils.devices import detect

from .base import ResultHandlerMixin, SessionsHandler, log_message
from .messages import Message
from .state import SessionState, SharedState

log = logging.getLogger(__name__)

class SessionHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def get(self):
        self.write({
            "status": "ok",
            "id": "suurrrree",
        })

    async def post(self):
        request_data = tornado.escape.json_decode(self.request.body) # why do you send text/plain from the client.
        if request_data['type'] == "restricted":
            # uuid is guessible. so we need a more random set. hash some random bytes. check flask or django
            id = str(uuid.uuid4()) # TODO: figure out if you wanna call it id or uuid
            # id = "2222222"
            self.state.register_session(id)
            self.set_secure_cookie("SESSID", id, expires_days=None)
            self.write({
                "status": "ok",
                "id": id,
            })

class SessionDatasetHandler(SessionsHandler, tornado.web.RequestHandler):
    async def get(self):
        self.write({
            "status": "ok",
            "id": "ok",
        })

    async def put(self, sess_id, uuid):
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

            self.state.dataset_state.register(
                uuid=uuid,
                dataset=ds,
                params=request_data['dataset'],
                converted=dataset_params,
            )
            details = await self.state.dataset_state.serialize(dataset_id=uuid)
            msg = Message(self.state).create_dataset(dataset=uuid, details=details)
            log_message(msg)
            self.write(msg)
            self.event_registry.broadcast_event(msg)
        except Exception as e:
            if uuid in self.state.dataset_state:
                await self.state.dataset_state.remove(uuid)
            msg = Message(self.state).create_dataset_error(uuid, str(e))
            log_message(msg, exception=True)
            self.write(msg)
            return
