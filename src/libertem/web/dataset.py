import time
import logging
import functools

import tornado.web
from opentelemetry import trace
import numpy as np

from libertem.io.dataset import load, detect, get_dataset_cls
from .base import CORSMixin, log_message
from libertem.common.async_utils import sync_to_async
from .messages import Message
from .state import SharedState

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


def prime_numba_cache(ds):
    dtypes = (np.float32, None)
    for dtype in dtypes:
        roi = np.zeros(ds.shape.nav, dtype=bool).reshape((-1,))
        roi[max(-ds._meta.sync_offset, 0)] = True

        from libertem.udf.sum import SumUDF
        from libertem.udf.raw import PickUDF
        from libertem.io.corrections.corrset import CorrectionSet
        from libertem.io.dataset.base import Negotiator

        # need to have at least one UDF; here we run for both sum and pick
        # to reduce the initial latency when switching to pick mode
        udfs = [SumUDF(), PickUDF()]
        neg = Negotiator()
        if ds.supports_correction():
            corr_dtypes = (np.float32, None)
        else:
            corr_dtypes = (None, )
        for udf in udfs:
            for corr_dtype in corr_dtypes:
                if corr_dtype is not None:
                    corrections = CorrectionSet(dark=np.zeros(ds.shape.sig, dtype=corr_dtype))
                else:
                    corrections = None
                found_first_tile = False
                for p in ds.get_partitions():
                    if found_first_tile:
                        break
                    p.set_corrections(corrections)
                    tiling_scheme = neg.get_scheme(
                        udfs=[udf],
                        dataset=ds,
                        approx_partition_shape=p.shape,
                        read_dtype=dtype,
                        roi=roi,
                        corrections=corrections,
                    )
                    for t in p.get_tiles(tiling_scheme=tiling_scheme, roi=roi):
                        found_first_tile = True
                        break


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
        msg = Message().delete_dataset(uuid)
        log_message(msg)
        self.event_registry.broadcast_event(msg)
        self.write(msg)

    async def prime_numba_caches(self, ds):
        executor = await self.state.executor_state.get_executor()

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
        request_data: dict = tornado.escape.json_decode(self.request.body)
        params = request_data['dataset']['params']
        params["type"] = ds_type = params["type"].upper()
        cls = get_dataset_cls(ds_type)
        ConverterCls = cls.get_msg_converter()
        converter = ConverterCls()
        try:
            dataset_params = converter.to_python(params)
            executor = await self.state.executor_state.get_executor()

            ds = await load(
                filetype=cls, executor=executor, enable_async=True, **dataset_params
            )

            with tracer.start_as_current_span("prime_numba_caches"):
                await self.prime_numba_caches(ds)

            self.dataset_state.register(
                uuid=uuid,
                dataset=ds,
                params=request_data['dataset'],
                converted=dataset_params,
            )
            details = await self.dataset_state.serialize(dataset_id=uuid)
            msg = Message().create_dataset(dataset=uuid, details=details)
            log_message(msg)
            self.write(msg)
            self.event_registry.broadcast_event(msg)
        except Exception as e:
            if uuid in self.dataset_state:
                await self.dataset_state.remove(uuid)
            msg = Message().create_dataset_error(uuid, str(e))
            log_message(msg, exception=True)
            self.write(msg)
            return


class DataSetDetectHandler(tornado.web.RequestHandler):
    def initialize(self, state: SharedState, event_registry):
        self.state = state
        self.event_registry = event_registry

    async def get(self):
        path = self.request.arguments['path'][0].decode("utf8")
        executor = await self.state.executor_state.get_executor()

        detected_params = await sync_to_async(
            detect, path=path, executor=executor.ensure_sync()
        )

        if not detected_params:
            msg = Message().dataset_detect_failed(path=path)
            log_message(msg)
            self.write(msg)
            return
        params = detected_params["parameters"]
        info = {}
        if "info" in detected_params:
            info = detected_params["info"]
        params.update({"type": detected_params["type"].upper()})
        info.update({"type": detected_params["type"].upper()})
        msg = Message().dataset_detect(params=params, info=info)
        log_message(msg)
        self.write(msg)
