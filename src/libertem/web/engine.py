import time
import logging
import asyncio
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

from opentelemetry import trace

from libertem.analysis.base import Analysis, AnalysisResultSet
from libertem.common.executor import JobCancelledError
from libertem.common.tracing import TracedThreadPoolExecutor
from libertem.common.progress import ProgressReporter, ProgressState
from libertem.udf.base import UDFResults
from libertem.io.dataset.base.dataset import DataSet
from libertem.web.models import AnalysisDetails
from .messages import Message
from .state import SharedState
from .base import log_message, result_images

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


if TYPE_CHECKING:
    from .events import EventRegistry

T = TypeVar('T')


class WebProgressReporter(ProgressReporter):
    """
    Progress reporter that pumps `ProgressState`
    messages out via the event registry mechanism.

    The `start`/`update`/`end` callbacks are invoked from
    the context of the `UDFRunner` implementation, so
    we need to make sure we schedule the websocket
    response tasks on the correct event loop
    (the one used by the web API)
    """

    def __init__(
        self,
        state: SharedState,
        event_registry: "EventRegistry",
        loop: asyncio.AbstractEventLoop,
        job_id: str,
    ):
        self.event_registry = event_registry
        self.state = state
        self.loop = loop
        self.job_id = job_id

    def start(self, state: ProgressState):
        self.handle_state_update(state, "start")

    def update(self, state: ProgressState):
        self.handle_state_update(state, "update")

    def end(self, state: ProgressState):
        self.handle_state_update(state, "end")

    def handle_state_update(self, state: ProgressState, event: str):
        msg = Message().job_progress(
            job_id=self.job_id, state=state, event=event
        )

        async def _task():
            await self.event_registry.broadcast_event(msg)
        asyncio.run_coroutine_threadsafe(_task(), loop=self.loop)


class JobEngine:
    def __init__(self, state: SharedState, event_registry: "EventRegistry"):
        self.state = state
        self.event_registry = event_registry
        self._pool = TracedThreadPoolExecutor(tracer=tracer)

    async def run_sync(self, fn: Callable[..., T], *args, **kwargs) -> T:
        # to make sure everything is using `self._pool`, only import
        # `sync_to_async` as a local:
        from libertem.common.async_utils import sync_to_async
        return await sync_to_async(fn, self._pool, *args, **kwargs)

    async def run_analysis(self, analysis_id: str, job_id: str):
        with tracer.start_as_current_span("JobEngine.run_analysis"):
            analysis_state = self.state.analysis_state[analysis_id]
            ds = self.state.dataset_state[analysis_state['dataset']]

            analysis_details = analysis_state["details"]
            analysis_type = analysis_details["analysisType"]
            params = analysis_details["parameters"]

            analysis = Analysis.get_analysis_by_type(analysis_type)(
                dataset=ds,
                parameters=params,
            )

            try:
                if analysis.TYPE != 'UDF':
                    raise TypeError(
                        'Only Analysis classes with TYPE="UDF" are supported'
                    )
                # Users of this class may call `register_job` before actually running it,
                # for example if they want to finish their response before starting to
                # actually run the UDF, so we check here and `register_job` if it hasn't
                # happened already:
                if job_id not in self.state.job_state:
                    await self.register_job(analysis_id, job_id)
                return await self.run_udf(
                    job_id=job_id,
                    dataset=ds,
                    dataset_id=analysis_state['dataset'],
                    analysis=analysis,
                    analysis_id=analysis_id,
                    details=analysis_details,
                )
            except JobCancelledError:
                msg = Message().cancel_done(job_id)
                log_message(msg)
                await self.event_registry.broadcast_event(msg)
                return
            except Exception as e:
                log.exception("error running job, params=%r", params)
                msg = Message().job_error(job_id, "error running job: %s" % str(e))
                self.event_registry.broadcast_event(msg)
                await self.state.job_state.remove(job_id)

    async def register_job(self, analysis_id: str, job_id: str):
        analysis_state = self.state.analysis_state[analysis_id]
        # FIXME: naming? job_state for UDFs?
        self.state.job_state.register(
            job_id=job_id, analysis_id=analysis_id, dataset_id=analysis_state['dataset'],
        )
        self.state.analysis_state.add_job(analysis_id, job_id)

    async def run_udf(
        self,
        job_id: str,
        dataset: DataSet,
        dataset_id: str,
        analysis: Analysis,
        analysis_id: str,
        details: AnalysisDetails,
    ) -> AnalysisResultSet:
        with tracer.start_as_current_span("JobEngine.run_udf"):
            return await self._run_udf(
                job_id,
                dataset,
                dataset_id,
                analysis,
                analysis_id,
                details,
            )

    async def _run_udf(
        self,
        job_id: str,
        dataset: DataSet,
        dataset_id: str,
        analysis: Analysis,
        analysis_id: str,
        details: AnalysisDetails,
    ) -> AnalysisResultSet:
        udf = analysis.get_udf()
        roi = analysis.get_roi()

        executor = await self.state.executor_state.get_executor()
        serialized_job = self.state.job_state.serialize(job_id)
        msg = Message().start_job(
            serialized_job=serialized_job, analysis_id=analysis_id,
        )
        self.event_registry.broadcast_event(msg)

        try:
            return await analysis.controller(  # type: ignore
                cancel_id=job_id, executor=executor,
                job_is_cancelled=lambda: self.state.job_state.is_cancelled(job_id),
                send_results=lambda results, finished: self.send_results(
                    results, job_id, finished=finished,
                    details=details, analysis_id=analysis_id,
                )
            )
        except NotImplementedError:
            pass

        # short circuit if the parameters only change the visualization
        # (as determined by the analysis via `Analysis.need_rerun`):
        if self.state.analysis_state.have_results(analysis_id):
            old_results = self.state.analysis_state.get_results(analysis_id)
            old_details, _, _, old_udf_results = old_results
            if not analysis.need_rerun(
                old_details["parameters"],
                details["parameters"],
            ):
                results = await self.run_sync(
                    analysis.get_udf_results,
                    udf_results=old_udf_results.buffers[0],
                    roi=roi,
                    damage=old_udf_results.damage
                )
                await self.send_results(
                    results, job_id, analysis_id, details, finished=True,
                    udf_results=old_udf_results,
                )
                return results

        t = time.time()
        post_t = time.time()
        window = 0.3
        # FIXME: allow to set correction data for a dataset via upload and local loading
        corrections = dataset.get_correction_data()
        runner_cls = executor.get_udf_runner()
        progress_reporter = WebProgressReporter(
            state=self.state,
            event_registry=self.event_registry,
            loop=asyncio.get_event_loop(),
            job_id=job_id,
        )
        result_iter = runner_cls(
            udfs=[udf],
            progress_reporter=progress_reporter
        ).run_for_dataset_async(
            dataset, executor,
            roi=roi,
            cancel_id=job_id,
            corrections=corrections,
            progress=True,
        )
        async for udf_results in result_iter:
            window = min(max(window, 2*(t - post_t)), 5)
            if time.time() - t < window:
                continue
            results = await self.run_sync(
                analysis.get_udf_results,
                udf_results=udf_results.buffers[0],
                roi=roi,
                damage=udf_results.damage
            )
            post_t = time.time()
            await self.send_results(
                results, job_id, analysis_id, details, udf_results=udf_results
            )
            # The broadcast might have taken quite some time due to
            # backpressure from the network
            t = time.time()

        if self.state.job_state.is_cancelled(job_id):
            raise JobCancelledError()
        results = await self.run_sync(
            analysis.get_udf_results,
            udf_results=udf_results.buffers[0],
            roi=roi,
            damage=udf_results.damage
        )
        await self.send_results(
            results, job_id, analysis_id, details, finished=True, udf_results=udf_results
        )
        return results

    async def send_results(
        self,
        results: AnalysisResultSet,
        job_id: str,
        analysis_id: str,
        details: AnalysisDetails,
        finished=False,
        udf_results: Optional[UDFResults] = None
    ) -> None:
        if self.state.job_state.is_cancelled(job_id):
            raise JobCancelledError()
        images = await result_images(results, self._pool)
        if self.state.job_state.is_cancelled(job_id):
            raise JobCancelledError()
        if finished:
            serialized_job = self.state.job_state.serialize(job_id)
            msg = Message().finish_job(
                serialized_job=serialized_job,
                num_images=len(results),
                image_descriptions=[
                    {"title": result.title, "desc": result.desc,
                    "includeInDownload": result.include_in_download}
                    for result in results
                ],
            )
            self.state.analysis_state.set_results(
                analysis_id, details, results, job_id, udf_results
            )
        else:
            msg = Message().task_result(
                job_id=job_id,
                num_images=len(results),
                image_descriptions=[
                    {"title": result.title, "desc": result.desc,
                    "includeInDownload": result.include_in_download}
                    for result in results
                ],
            )
        log_message(msg)
        # NOTE: make sure the following broadcast_event messages are sent atomically!
        # (that is: keep the code below synchronous, and only send the messages
        # once the images have finished encoding, and then send all at once)
        futures = []
        futures.append(
            self.event_registry.broadcast_event(msg)
        )
        for image in images:
            raw_bytes = image.read()
            futures.append(
                self.event_registry.broadcast_event(raw_bytes, binary=True)
            )
        await asyncio.gather(*futures)

    async def send_existing_job_results(self):
        results = self.state.analysis_state.get_all_results()
        for analysis_id, (details, result_set, job_id, udf_results) in results:
            serialized_job = self.state.job_state.serialize(job_id)
            await self.event_registry.broadcast_event(
                Message().start_job(
                    serialized_job=serialized_job, analysis_id=analysis_id,
                )
            )
            await self.send_results(
                result_set, job_id, analysis_id, details, finished=True, udf_results=udf_results
            )
