import logging

import tornado.web

from libertem.io.dataset import load, detect, get_dataset_cls
from libertem.io.dataset.base import Negotiator
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

    async def prime_numba_caches(self, ds):
        executor = self.state.executor_state.get_executor()

        def log_stuff(fn):
            def _inner(dask_worker):
                print(f"running on {dask_worker}")
                return fn()
            return _inner

        def _prime_cache():
            import numpy as np
            dtypes = (np.float32, np.float64, None)
            for dtype in dtypes:
                roi = np.zeros(ds.shape, dtype=np.bool).reshape((-1,))
                roi[0] = 1

                from libertem.udf.sum import SumUDF
                from libertem.corrections.corrset import CorrectionSet

                udfs = [SumUDF()]  # need to have at least one UDF
                p = next(ds.get_partitions())
                neg = Negotiator()
                for corr_dtype in (np.float32, np.float64):
                    corrections = CorrectionSet(dark=np.zeros(ds.shape.sig, dtype=corr_dtype))
                    p.set_corrections(corrections)
                    tiling_scheme = neg.get_scheme(
                        udfs=udfs,
                        partition=p,
                        read_dtype=dtype,
                        roi=roi,
                        corrections=corrections,
                    )
                    next(p.get_tiles(tiling_scheme=tiling_scheme))

        # first: make sure the jited functions used for I/O are compiled
        # by running a single-core workload on each host:
        await executor.run_each_host(_prime_cache)

        # second: make sure each worker *process* has the jited functions
        # loaded from the cache
        await executor.run_each_worker(log_stuff(_prime_cache))

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

        detected_params = await run_blocking(
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
