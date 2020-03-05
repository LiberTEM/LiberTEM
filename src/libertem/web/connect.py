import logging
from functools import partial

import tornado.web

from libertem.executor.base import AsyncAdapter, sync_to_async
from libertem.executor.dask import DaskJobExecutor
from .messages import Message
from .base import log_message
from .state import SharedState

log = logging.getLogger(__name__)


class ConnectHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def get(self):
        log.info("ConnectHandler.get")
        try:
            self.state.executor_state.get_executor()
            params = self.state.executor_state.get_cluster_params()
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
                "local_directory": self.state.get_local_directory()
            }
            if "numWorkers" in connection:
                cluster_kwargs.update({"n_workers": connection["numWorkers"]})
            sync_executor = await sync_to_async(
                partial(DaskJobExecutor.make_local, cluster_kwargs=cluster_kwargs)
            )
        else:
            raise ValueError("unknown connection type")
        executor = AsyncAdapter(wrapped=sync_executor)
        await self.state.executor_state.set_executor(executor, request_data)
        await self.state.dataset_state.verify()
        datasets = await self.state.dataset_state.serialize_all()
        msg = Message(self.state).initial_state(
            jobs=self.state.job_state.serialize_all(),
            datasets=datasets, analyses=self.state.analysis_state.serialize_all(),
        )
        log_message(msg)
        # FIXME: don't broadcast, only send to the websocket that matches this HTTP connection
        # (is this even possible?)
        self.event_registry.broadcast_event(msg)
        self.write({
            "status": "ok",
            "connection": connection,
        })
