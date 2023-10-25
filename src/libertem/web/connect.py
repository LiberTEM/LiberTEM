import logging
from functools import partial
from typing import TYPE_CHECKING, Dict, List, Any, Tuple

import tornado.web
from opentelemetry import trace

from libertem.executor.base import AsyncAdapter
from libertem.common.async_utils import sync_to_async
from libertem.executor.dask import DaskJobExecutor, cluster_spec
from libertem.web.engine import JobEngine
from .messages import Message
from .base import log_message
from .state import SharedState
from libertem.utils.devices import detect

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

if TYPE_CHECKING:
    from libertem.web.events import EventRegistry


def _int_or_zero(value) -> int:
    try:
        return int(value)
    except ValueError:
        return 0


def _convert_device_map(raw_cudas: Dict[int, Any]) -> List[int]:
    return [
        this_id
        for dev_id, num in raw_cudas.items()
        for this_id in [dev_id]*_int_or_zero(num)
    ]


def create_executor(*, connection, local_directory, preload) -> DaskJobExecutor:
    devices = detect()
    options = {
        "local_directory": local_directory,
    }
    if "numWorkers" in connection:
        num_workers = connection["numWorkers"]
        if not isinstance(num_workers, int) or num_workers < 1:
            raise ValueError('Number of workers must be positive integer')
        devices["cpus"] = range(num_workers)
    raw_cudas = connection.get("cudas", {})
    cudas = _convert_device_map(raw_cudas)
    devices["cudas"] = cudas
    return DaskJobExecutor.make_local(
        spec=cluster_spec(
            **devices,
            options=options,
            preload=preload,
        )
    )


def create_executor_external(
    executor_spec: Dict[str, int],
    local_directory,
    preload,
) -> Tuple[AsyncAdapter, Dict[str, Dict[str, Any]]]:
    cudas = {}
    if executor_spec['cudas']:
        cudas[0] = executor_spec['cudas']
    params = {
        "connection": {
            "type": "LOCAL",
            "numWorkers": executor_spec['cpus'],
            "cudas": cudas,
        }
    }
    sync_executor = create_executor(
        connection=params['connection'],
        local_directory=local_directory,
        preload=preload
    )
    pool = AsyncAdapter.make_pool()
    executor = AsyncAdapter(wrapped=sync_executor, pool=pool)
    return executor, params


class ConnectHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry: "EventRegistry"):
        self.state = state
        self.event_registry = event_registry
        self.engine = JobEngine(state, event_registry)

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
        with tracer.start_as_current_span("ConnectHandler.put"):
            await self._do_connect()

    async def _do_connect(self):
        # TODO: extract json request data stuff into mixin?
        request_data = tornado.escape.json_decode(self.request.body)
        connection = request_data['connection']
        pool = AsyncAdapter.make_pool()
        with tracer.start_as_current_span("executor setup"):
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
                try:
                    sync_executor = await sync_to_async(
                        create_executor,
                        pool=pool,
                        connection=connection,
                        local_directory=self.state.get_local_directory(),
                        preload=self.state.get_preload(),
                    )
                except Exception as e:
                    msg = Message(self.state).cluster_conn_error(msg=str(e))
                    log_message(msg)
                    self.write(msg)
                    return None
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
        await self.engine.send_existing_job_results()
        self.write({
            "status": "ok",
            "connection": connection,
        })
