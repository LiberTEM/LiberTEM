import os
import copy
import typing
import itertools
import logging
import asyncio
import time
import contextlib

import psutil

import libertem
from libertem.api import Context
from libertem.analysis.base import AnalysisResultSet
from libertem.common.executor import JobExecutor
from libertem.common.async_utils import sync_to_async
from libertem.executor.base import AsyncAdapter
from libertem.executor.dask import DaskJobExecutor
from libertem.io.dataset.base import DataSetException, DataSet
from libertem.io.writers.results.base import ResultFormatRegistry
from libertem.io.writers.results import formats  # NOQA
from libertem.udf.base import UDFResults
from libertem.utils import devices
from .models import (
    AnalysisDetails, AnalysisInfo, AnalysisResultInfo, CompoundAnalysisInfo, AnalysisParameters,
    DatasetInfo, JobInfo, SerializedJobInfo,
)
from .helpers import create_executor
from .event_bus import EventBus

log = logging.getLogger(__name__)


class ExecutorState:
    """
    Executor management for the web API. This class is used by `SharedState` to
    manage executor instances and related state.  Executors passed into
    `set_executors` will be taken over, and managed directly by this class.

    Be sure to explicitly call the `shutdown` method to clean up, otherwise
    there may be dangling resources preventing a clean shutdown.
    """
    def __init__(
        self,
        event_bus: EventBus,
        loop: typing.Optional[asyncio.AbstractEventLoop] = None,
        snooze_timeout: typing.Optional[float] = None,
    ):
        self.executor = None
        self.cluster_params = {}
        self.cluster_details: typing.Optional[list] = None
        self.context: typing.Optional[Context] = None
        self._event_bus = event_bus
        self._snooze_lock = asyncio.Lock()
        self._snooze_timeout = snooze_timeout
        self._pool = AsyncAdapter.make_pool()
        self._is_snoozing = False
        self._last_activity = time.monotonic()
        self._snooze_check_interval = min(
            30.0,
            snooze_timeout and (snooze_timeout * 0.1) or 30.0,
        )
        self._snooze_task = asyncio.ensure_future(self._snooze_check_task())
        if loop is None:
            loop = asyncio.get_event_loop()
        self._loop = loop
        self._keep_alive = 0
        self.local_directory = "dask-worker-space"
        self.preload: tuple[str, ...] = ()

    def set_preload(self, preload: tuple[str, ...]) -> None:
        self.preload = preload

    def get_preload(self) -> tuple[str, ...]:
        return self.preload

    def set_local_directory(self, local_directory: str) -> None:
        if local_directory is not None:
            self.local_directory = local_directory

    def get_local_directory(self):
        return self.local_directory

    def _update_last_activity(self):
        self._last_activity = time.monotonic()
        log.debug("_update_last_activity")

    @contextlib.contextmanager
    def keep_alive(self):
        self._update_last_activity()
        self._keep_alive += 1
        try:
            yield
        finally:
            self._keep_alive -= 1
            self._update_last_activity()

    async def snooze(self):
        async with self._snooze_lock:
            if self.executor is None:
                log.debug("not snoozing: no executor")
                return
            if self._keep_alive > 0:
                log.debug("not snoozing: _keep_alive=%d", self._keep_alive)
                return
            log.info("Snoozing...")
            from .messages import Message
            msg = Message().snooze("snoozing")
            self._event_bus.send(msg)
            self.executor.ensure_sync().close()
            self.context = None
            self.executor = None
            self._is_snoozing = True

    async def unsnooze(self):
        async with self._snooze_lock:
            if not self._is_snoozing:
                return
            log.info("Unsnoozing...")
            from .messages import Message
            msg = Message().unsnooze("unsnoozing")
            self._event_bus.send(msg)
            executor = await self.make_executor(self.cluster_params, self._pool)
            self._set_executor(executor, self.cluster_params)
            self._is_snoozing = False
            msg = Message().unsnooze_done("unsnooze done")
            self._event_bus.send(msg)

    def is_snoozing(self):
        return self._is_snoozing

    async def _snooze_check_task(self):
        """
        Periodically check if we need to snooze the executor
        """
        try:
            while True:
                await asyncio.sleep(self._snooze_check_interval)
                if not self._is_snoozing and self._snooze_timeout is not None:
                    since_last_activity = time.monotonic() - self._last_activity
                    if since_last_activity > self._snooze_timeout:
                        await self.snooze()
        except asyncio.CancelledError:
            log.debug("shutting down snooze check task")
            return

    async def make_executor(self, params, pool) -> AsyncAdapter:
        connection = params['connection']
        if connection["type"].lower() == "tcp":
            sync_executor = await sync_to_async(
                DaskJobExecutor.connect,
                pool=self._pool,
                scheduler_uri=connection['address'],
            )
        elif connection["type"].lower() == "local":
            sync_executor = await sync_to_async(
                create_executor,
                pool=self._pool,
                connection=connection,
                local_directory=self.get_local_directory(),
                preload=self.get_preload(),
            )
        else:
            raise ValueError("unknown connection type")
        executor = AsyncAdapter(wrapped=sync_executor, pool=pool)
        return executor

    async def get_executor(self):
        self._update_last_activity()
        await self.unsnooze()
        if self.executor is None:
            # TODO: exception type, conversion into 400 response
            raise RuntimeError("wrong state: executor is None")
        return self.executor

    def have_executor(self):
        return self.executor is not None or self._is_snoozing

    async def get_resource_details(self):
        # memoize the cluster details, if ever we support
        # dynamic resources this will need to change
        if self.cluster_details is None:
            executor = await self.get_executor()
            self.cluster_details = await executor.get_resource_details()
        return self.cluster_details

    async def get_context(self) -> Context:
        await self.unsnooze()
        self._update_last_activity()
        if self.context is None:
            raise RuntimeError("cannot get context, please call `set_executor` before")
        return self.context

    def shutdown(self):
        self._loop.call_soon_threadsafe(self._snooze_task.cancel)
        if self.context is not None:
            self.context.close()

    async def set_executor(self, executor: JobExecutor, params):
        """
        Set the new executor used to run jobs, and the parameters used
        to create/connect.

        After the executor is set, we take "ownership" of it, and ensure
        that it is properly cleaned up in the `shutdown` method.
        """
        self._update_last_activity()
        if self.executor is not None:
            await self.executor.close()
            self.executor = None
        self._set_executor(executor, params)

    def _set_executor(self, executor: JobExecutor, params):
        if self.executor is not None:
            self.executor.ensure_sync().close()
        self.executor = executor
        self.cluster_params = params
        self.context = Context(executor=executor.ensure_sync())
        self._is_snoozing = False  # if we were snoozing before, we no longer are
        try:
            # Exposing the scheduler address allows use of
            # libertem-server to spin up a LT-compatible
            # persistent dask cluster
            log.info(f'Scheduler at {self.context.executor.client.scheduler.address}')
        except AttributeError:
            pass

    def get_cluster_params(self):
        self._update_last_activity()
        return self.cluster_params


class AnalysisState:
    def __init__(self, executor_state: ExecutorState, job_state: 'JobState'):
        self.analyses: dict[str, AnalysisInfo] = {}
        self.results: dict[str, AnalysisResultInfo] = {}
        self.job_state = job_state

    def create(
        self, uuid: str, dataset_uuid: str, analysis_type: str, parameters: AnalysisParameters
    ) -> None:
        assert uuid not in self.analyses
        self.analyses[uuid] = {
            "dataset": dataset_uuid,
            "analysis": uuid,
            "jobs": [],
            "details": {
                "analysisType": analysis_type,
                "parameters": parameters,
            },
        }

    def add_job(self, analysis_id: str, job_id: str) -> None:
        jobs = self.analyses[analysis_id]["jobs"]
        jobs.append(job_id)

    def update(self, uuid: str, analysis_type: str, parameters: AnalysisParameters) -> None:
        self.analyses[uuid]["details"]["parameters"] = parameters
        self.analyses[uuid]["details"]["analysisType"] = analysis_type

    def get(
        self, uuid: str, default: typing.Optional[AnalysisInfo] = None
    ) -> typing.Optional[AnalysisInfo]:
        return self.analyses.get(uuid, default)

    def filter(self, predicate: typing.Callable[[AnalysisInfo], bool]) -> list[AnalysisInfo]:
        return [
            analysis
            for analysis in self.analyses.values()
            if predicate(analysis)
        ]

    async def remove(self, uuid: str) -> bool:
        if uuid not in self.analyses:
            return False
        if uuid in self.results:
            self.remove_results(uuid)
        await self.remove_jobs(uuid)
        del self.analyses[uuid]
        return True

    async def remove_jobs(self, uuid: str) -> None:
        jobs = copy.copy(self.job_state.get_for_analysis_id(uuid))
        for job_id in jobs:
            await self.job_state.remove(job_id)

    def remove_results(self, uuid: str) -> None:
        del self.results[uuid]

    def set_results(
        self,
        analysis_id: str,
        details: AnalysisDetails,
        results: AnalysisResultSet,
        job_id: str,
        udf_results: typing.Optional[UDFResults],
    ) -> None:
        """
        set results: create or update
        """
        self.results[analysis_id] = AnalysisResultInfo(
            copy.deepcopy(details), results, job_id, udf_results
        )

    def have_results(self, analysis_id: str) -> bool:
        return analysis_id in self.results

    def get_results(self, analysis_id: str) -> AnalysisResultInfo:
        return self.results[analysis_id]

    def get_all_results(self) -> typing.ItemsView[str, AnalysisResultInfo]:
        return self.results.items()

    def __getitem__(self, analysis_id: str) -> AnalysisInfo:
        return self.analyses[analysis_id]

    def serialize(self, analysis_id: str) -> AnalysisInfo:
        result = copy.copy(self[analysis_id])
        result["jobs"] = [
            job_id
            for job_id in result["jobs"]
            if not self.job_state.is_cancelled(job_id)
        ]
        return result

    def serialize_all(self) -> list[AnalysisInfo]:
        return [
            self.serialize(analysis_id)
            for analysis_id in self.analyses
        ]


class CompoundAnalysisState:
    def __init__(self, analysis_state: AnalysisState):
        self.analysis_state = analysis_state
        self.analyses: dict[str, CompoundAnalysisInfo] = {}

    def create_or_update(
        self, uuid: str, main_type: str, dataset_id: str, analyses: list[str]
    ) -> bool:
        created = uuid not in self.analyses
        self.analyses[uuid] = {
            "dataset": dataset_id,
            "compoundAnalysis": uuid,
            "details": {
                "mainType": main_type,
                "analyses": analyses,
            }
        }
        return created

    def remove(self, uuid: str) -> None:
        del self.analyses[uuid]

    def __getitem__(self, uuid: str) -> CompoundAnalysisInfo:
        return self.analyses[uuid]

    def filter(
        self, predicate: typing.Callable[[CompoundAnalysisInfo], bool]
    ) -> list[CompoundAnalysisInfo]:
        return [
            ca
            for ca in self.analyses.values()
            if predicate(ca)
        ]

    def serialize(self, uuid: str) -> CompoundAnalysisInfo:
        return self[uuid]

    def serialize_all(self) -> list[CompoundAnalysisInfo]:
        return [
            self.serialize(uuid)
            for uuid in self.analyses
        ]


class DatasetState:
    def __init__(self, executor_state: ExecutorState, analysis_state: AnalysisState,
                 compound_analysis_state: CompoundAnalysisState):
        self.datasets: dict[str, DatasetInfo] = {}
        self.dataset_to_id: dict[DataSet, str] = {}
        self.executor_state = executor_state
        self.analysis_state = analysis_state
        self.compound_analysis_state = compound_analysis_state

    def register(
        self, uuid: str, dataset: DataSet, params: dict, converted: dict
    ):
        assert uuid not in self.datasets
        self.datasets[uuid] = {
            "dataset": dataset,
            "params": params,
            "converted": converted,
        }
        self.dataset_to_id[dataset] = uuid
        return self

    async def serialize(self, dataset_id):
        executor = await self.executor_state.get_executor()
        dataset = self.datasets[dataset_id]
        diag = await executor.run_function(lambda: dataset["dataset"].diagnostics)
        return {
            "id": dataset_id,
            "params": {
                **dataset["params"]["params"],
                "shape": tuple(dataset["dataset"].shape),
            },
            "diagnostics": diag,
        }

    async def serialize_all(self):
        return [
            await self.serialize(dataset_id)
            for dataset_id in self.datasets.keys()
        ]

    def id_for_dataset(self, dataset):
        return self.dataset_to_id[dataset]

    def __getitem__(self, uuid):
        return self.datasets[uuid]["dataset"]

    def __contains__(self, uuid):
        return uuid in self.datasets

    async def verify(self):
        executor = await self.executor_state.get_executor()
        for uuid, params in self.datasets.items():
            dataset = params["dataset"]
            try:
                await executor.run_function(dataset.check_valid)
            except DataSetException:
                await self.remove_dataset(uuid)

    async def remove(self, uuid):
        """
        Stop all jobs and remove dataset state for the dataset identified by `uuid`
        """
        ds = self.datasets[uuid]["dataset"]
        analyses = self.analysis_state.filter(lambda a: a["dataset"] == uuid)
        compound_analyses = self.compound_analysis_state.filter(lambda ca: ca["dataset"] == uuid)
        del self.datasets[uuid]
        del self.dataset_to_id[ds]
        for analysis in analyses:
            await self.analysis_state.remove(analysis["analysis"])
        for ca in compound_analyses:
            self.compound_analysis_state.remove(ca["compoundAnalysis"])


class JobState:
    def __init__(self, executor_state: ExecutorState):
        self.jobs: dict[str, JobInfo] = {}
        self.executor_state = executor_state
        self.jobs_for_dataset = typing.DefaultDict[str, set[str]](lambda: set())
        self.jobs_for_analyses = typing.DefaultDict[str, set[str]](lambda: set())

    def register(self, job_id: str, analysis_id, dataset_id):
        assert job_id not in self.jobs
        self.jobs[job_id] = {
            "id": job_id,
            "analysis": analysis_id,
            "dataset": dataset_id,
        }
        self.jobs_for_dataset[dataset_id].add(job_id)
        self.jobs_for_analyses[analysis_id].add(job_id)
        return self

    async def remove(self, uuid: str) -> bool:
        try:
            executor = await self.executor_state.get_executor()
            await executor.cancel(uuid)
            del self.jobs[uuid]
            for ds, jobs in itertools.chain(self.jobs_for_dataset.items(),
                                            self.jobs_for_analyses.items()):
                if uuid in jobs:
                    jobs.remove(uuid)
            return True
        except KeyError:
            return False

    def get_for_dataset_id(self, dataset_id: str) -> set[str]:
        return self.jobs_for_dataset[dataset_id]

    def get_for_analysis_id(self, analysis_id: str) -> set[str]:
        return self.jobs_for_analyses[analysis_id]

    def __getitem__(self, uuid: str) -> JobInfo:
        return self.jobs[uuid]

    def __contains__(self, uuid: str) -> bool:
        return uuid in self.jobs

    def is_cancelled(self, uuid: str) -> bool:
        return uuid not in self.jobs

    def serialize(self, job_id: str) -> SerializedJobInfo:
        job = self[job_id]
        return {
            "id": job["id"],
            "analysis": job["analysis"],
        }

    def serialize_all(self) -> list[SerializedJobInfo]:
        return [
            self.serialize(job_id)
            for job_id in self.jobs.keys()
        ]


class SharedState:
    def __init__(self, executor_state: "ExecutorState"):
        self.executor_state = executor_state
        self.job_state = JobState(self.executor_state)
        self.analysis_state = AnalysisState(self.executor_state, job_state=self.job_state)
        self.compound_analysis_state = CompoundAnalysisState(self.analysis_state)
        self.dataset_state = DatasetState(
            self.executor_state,
            analysis_state=self.analysis_state,
            compound_analysis_state=self.compound_analysis_state,
        )

    def get_local_cores(self, default: int = 2) -> int:
        cores: typing.Optional[int] = psutil.cpu_count(logical=False)
        if cores is None:
            cores = default
        return cores

    def get_ds_type_info(self, ds_type_id: str):
        from libertem.io.dataset import get_dataset_cls
        cls = get_dataset_cls(ds_type_id)
        ConverterCls = cls.get_msg_converter()
        converter = ConverterCls()
        schema = converter.SCHEMA
        supported_backends = cls.get_supported_io_backends()
        default_backend = cls.get_default_io_backend().id_
        if not supported_backends:
            default_backend = None
        return {
            "schema": schema,
            "default_io_backend": default_backend,
            "supported_io_backends": supported_backends,
        }

    def get_config(self):
        from libertem.io.dataset import filetypes
        detected_devices = devices.detect()
        ds_types = list(filetypes.keys())
        return {
            "version": libertem.__version__,
            "resultFileFormats": ResultFormatRegistry.get_available_formats(),
            "revision": libertem.revision,
            "localCores": self.get_local_cores(),
            "devices": detected_devices,
            "datasetTypes": {
                ds_type_id.upper(): self.get_ds_type_info(ds_type_id)
                for ds_type_id in ds_types
            },
            "cwd": os.getcwd(),
            # '/' works on Windows, too.
            "separator": '/'
        }

    async def create_and_set_executor(self, spec: dict[str, int]):
        """
        Create a new executor from spec, a dict[str, int]
        compatible with the main arguments of cluster_spec().
        Any values not in spec are filled from a call to detect()

        Any existing executor will first closed by the call
        to self.executor_state._set_executor
        """
        from .helpers import create_executor_external  # circular import
        executor, params = create_executor_external(
            spec,
            self.executor_state.get_local_directory(),
            self.executor_state.get_preload(),
        )
        self.executor_state._set_executor(executor, params)
