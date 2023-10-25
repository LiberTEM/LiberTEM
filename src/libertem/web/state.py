import os
import copy
import typing
import itertools
import logging

import psutil

import libertem
from libertem.api import Context
from libertem.analysis.base import AnalysisResultSet
from libertem.common.executor import JobExecutor
from libertem.io.dataset.base import DataSetException, DataSet
from libertem.io.writers.results.base import ResultFormatRegistry
from libertem.io.writers.results import formats  # NOQA
from libertem.udf.base import UDFResults
from libertem.utils import devices
from .models import (
    AnalysisDetails, AnalysisInfo, AnalysisResultInfo, CompoundAnalysisInfo, AnalysisParameters,
    DatasetInfo, JobInfo, SerializedJobInfo,
)

log = logging.getLogger(__name__)


class ExecutorState:
    def __init__(self):
        self.executor = None
        self.cluster_params = {}
        self.context: typing.Optional[Context] = None

    def get_executor(self):
        if self.executor is None:
            # TODO: exception type, conversion into 400 response
            raise RuntimeError("wrong state: executor is None")
        return self.executor

    def have_executor(self):
        return self.executor is not None

    def get_context(self) -> Context:
        if self.context is None:
            raise RuntimeError("cannot get context, please call `set_executor` before")
        return self.context

    async def set_executor(self, executor: JobExecutor, params):
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
        try:
            # Exposing the scheduler address allows use of
            # libertem-server to spin up a LT-compatible
            # persistent dask cluster
            log.info(f'Scheduler at {self.context.executor.client.scheduler.address}')
        except AttributeError:
            pass

    def get_cluster_params(self):
        return self.cluster_params


class AnalysisState:
    def __init__(self, executor_state: ExecutorState, job_state: 'JobState'):
        self.analyses: typing.Dict[str, AnalysisInfo] = {}
        self.results: typing.Dict[str, AnalysisResultInfo] = {}
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

    def filter(self, predicate: typing.Callable[[AnalysisInfo], bool]) -> typing.List[AnalysisInfo]:
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

    def serialize_all(self) -> typing.List[AnalysisInfo]:
        return [
            self.serialize(analysis_id)
            for analysis_id in self.analyses
        ]


class CompoundAnalysisState:
    def __init__(self, analysis_state: AnalysisState):
        self.analysis_state = analysis_state
        self.analyses: typing.Dict[str, CompoundAnalysisInfo] = {}

    def create_or_update(
        self, uuid: str, main_type: str, dataset_id: str, analyses: typing.List[str]
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
    ) -> typing.List[CompoundAnalysisInfo]:
        return [
            ca
            for ca in self.analyses.values()
            if predicate(ca)
        ]

    def serialize(self, uuid: str) -> CompoundAnalysisInfo:
        return self[uuid]

    def serialize_all(self) -> typing.List[CompoundAnalysisInfo]:
        return [
            self.serialize(uuid)
            for uuid in self.analyses
        ]


class DatasetState:
    def __init__(self, executor_state: ExecutorState, analysis_state: AnalysisState,
                 compound_analysis_state: CompoundAnalysisState):
        self.datasets: typing.Dict[str, DatasetInfo] = {}
        self.dataset_to_id: typing.Dict[DataSet, str] = {}
        self.executor_state = executor_state
        self.analysis_state = analysis_state
        self.compound_analysis_state = compound_analysis_state

    def register(
        self, uuid: str, dataset: DataSet, params: typing.Dict, converted: typing.Dict
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
        executor = self.executor_state.get_executor()
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
        executor = self.executor_state.get_executor()
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
        self.jobs: typing.Dict[str, JobInfo] = {}
        self.executor_state = executor_state
        self.jobs_for_dataset = typing.DefaultDict[str, typing.Set[str]](lambda: set())
        self.jobs_for_analyses = typing.DefaultDict[str, typing.Set[str]](lambda: set())

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
            executor = self.executor_state.get_executor()
            await executor.cancel(uuid)
            del self.jobs[uuid]
            for ds, jobs in itertools.chain(self.jobs_for_dataset.items(),
                                            self.jobs_for_analyses.items()):
                if uuid in jobs:
                    jobs.remove(uuid)
            return True
        except KeyError:
            return False

    def get_for_dataset_id(self, dataset_id: str) -> typing.Set[str]:
        return self.jobs_for_dataset[dataset_id]

    def get_for_analysis_id(self, analysis_id: str) -> typing.Set[str]:
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

    def serialize_all(self) -> typing.List[SerializedJobInfo]:
        return [
            self.serialize(job_id)
            for job_id in self.jobs.keys()
        ]


class SharedState:
    def __init__(self):
        self.executor_state = ExecutorState()
        self.job_state = JobState(self.executor_state)
        self.analysis_state = AnalysisState(self.executor_state, job_state=self.job_state)
        self.compound_analysis_state = CompoundAnalysisState(self.analysis_state)
        self.dataset_state = DatasetState(
            self.executor_state,
            analysis_state=self.analysis_state,
            compound_analysis_state=self.compound_analysis_state,
        )
        self.local_directory = "dask-worker-space"
        self.preload: typing.Tuple[str, ...] = ()

    def get_local_cores(self, default: int = 2) -> int:
        cores: typing.Optional[int] = psutil.cpu_count(logical=False)
        if cores is None:
            cores = default
        return cores

    def set_local_directory(self, local_directory: str) -> None:
        if local_directory is not None:
            self.local_directory = local_directory

    def get_local_directory(self):
        return self.local_directory

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

    def set_preload(self, preload: typing.Tuple[str, ...]) -> None:
        self.preload = preload

    def get_preload(self) -> typing.Tuple[str, ...]:
        return self.preload

    def create_and_set_executor(self, spec: typing.Dict[str, int]):
        """
        Create a new executor from spec, a dict[str, int]
        compatible with the main arguments of cluster_spec().
        Any values not in spec are filled from a call to detect()

        Any existing executor will first closed by the call
        to self.executor_state._set_executor
        """
        from .connect import create_executor  # circular import
        executor, params = create_executor(
            spec,
            self.get_local_directory(),
            self.get_preload(),
        )
        self.executor_state._set_executor(executor, params)
