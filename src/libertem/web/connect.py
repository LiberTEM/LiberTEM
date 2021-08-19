import logging
from functools import partial

import tornado.web

from libertem.executor.base import AsyncAdapter, sync_to_async
from libertem.executor.dask import DaskJobExecutor, cluster_spec
from .messages import Message
from .base import log_message, ResultHandlerMixin
from .state import SharedState
from libertem.utils.devices import detect

log = logging.getLogger(__name__)


class ConnectHandler(ResultHandlerMixin, tornado.web.RequestHandler):
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
        pool = AsyncAdapter.make_pool()
        if connection["type"].lower() == "tcp":
            try:
                sync_executor = await sync_to_async(partial(DaskJobExecutor.connect,
                    scheduler_uri=connection['address'],
                ), pool=pool)
            except Exception as e:
                msg = Message(self.state).cluster_conn_error(msg=str(e))
                log_message(msg)
                self.write(msg)
                return None
        elif connection["type"].lower() == "local":
            devices = detect()
            options = {
                "local_directory": self.state.get_local_directory()
            }
            if "numWorkers" in connection:
                devices["cpus"] = range(connection["numWorkers"])
            devices["cudas"] = connection.get("cudas", [])

            sync_executor = await sync_to_async(partial(DaskJobExecutor.make_local,
                spec=cluster_spec(**devices, options=options)
            ), pool=pool)
        else:
            raise ValueError("unknown connection type")
        executor = AsyncAdapter(wrapped=sync_executor, pool=pool)
        await self.state.executor_state.set_executor(executor, request_data)
        await self.state.dataset_state.verify()
        datasets = await self.state.dataset_state.serialize_all()
        msg = Message(self.state).initial_state(
            jobs=self.state.job_state.serialize_all(),
            datasets=datasets, analyses=self.state.analysis_state.serialize_all(),
            compound_analyses=self.state.compound_analysis_state.serialize_all(),
        )
        log_message(msg)
        # FIXME: don't broadcast, only send to the websocket that matches this HTTP connection
        # (is this even possible?)
        self.event_registry.broadcast_event(msg)
        await self.send_existing_job_results()
        self.write({
            "status": "ok",
            "connection": connection,
        })
