import logging
from functools import partial

import tornado.web

from libertem.executor.base import AsyncAdapter, sync_to_async
from libertem.executor.dask import DaskJobExecutor
from .messages import Message
from .base import log_message

log = logging.getLogger(__name__)


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
            sync_executor = await sync_to_async(partial(DaskJobExecutor.connect,
                scheduler_uri=connection['address'],
            ))
        elif connection["type"].lower() == "local":
            cluster_kwargs = {
                "threads_per_worker": 1,
                "local_dir": self.data.get_local_directory()
            }
            if "numWorkers" in connection:
                cluster_kwargs.update({"n_workers": connection["numWorkers"]})
            sync_executor = await sync_to_async(
                partial(DaskJobExecutor.make_local, cluster_kwargs=cluster_kwargs)
            )
        else:
            raise ValueError("unknown connection type")
        executor = AsyncAdapter(wrapped=sync_executor)
        await self.data.set_executor(executor, request_data)
        await self.data.verify_datasets()
        datasets = await self.data.serialize_datasets()
        msg = Message(self.data).initial_state(
            jobs=self.data.serialize_jobs(),
            datasets=datasets,
        )
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write({
            "status": "ok",
            "connection": connection,
        })
