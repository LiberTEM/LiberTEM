import os
import copy
import typing
import itertools
from collections import defaultdict

import psutil

import libertem
from libertem.analysis.base import AnalysisResultSet
from libertem.io.dataset.base import DataSetException
from libertem.io.writers.results.base import ResultFormatRegistry
from libertem.io.writers.results import formats  # NOQA
from libertem.utils import devices


class ExecutorState:
    def __init__(self):
        self.executor = None
        self.cluster_params = {}

    def get_executor(self):
        if self.executor is None:
            # TODO: exception type, conversion into 400 response
            raise RuntimeError("wrong state: executor is None")
        return self.executor

    def have_executor(self):
        return self.executor is not None

    async def set_executor(self, executor, params):
        if self.executor is not None:
            await self.executor.close()
        self.executor = executor
        self.cluster_params = params

    def get_cluster_params(self):
        return self.cluster_params


AnalysisParameters = typing.Dict


class AnalysisState:
    def __init__(self, executor_state: ExecutorState, job_state: 'JobState'):
        self.analyses: typing.Dict[str, AnalysisParameters] = {}
        self.results: typing.Dict[str, typing.Tuple[AnalysisResultSet, AnalysisParameters]] = {}
        self.job_state = job_state

    def create(self, uuid, dataset_uuid, analysis_type, parameters):
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

    def add_job(self, analysis_id, job_id):
        jobs = self.analyses[analysis_id]["jobs"]
        jobs.append(job_id)

    def update(self, uuid, analysis_type, parameters):
        self.analyses[uuid]["details"]["parameters"] = parameters
        self.analyses[uuid]["details"]["analysisType"] = analysis_type

    def get(self, uuid, default=None):
        return self.analyses.get(uuid, default)

    def filter(self, predicate):
        return [
            analysis
            for analysis in self.analyses.values()
            if predicate(analysis)
        ]

    async def remove(self, uuid):
        if uuid not in self.analyses:
            return False
        if uuid in self.results:
            self.remove_results(uuid)
        await self.remove_jobs(uuid)
        del self.analyses[uuid]
        return True

    async def remove_jobs(self, uuid):
        jobs = copy.copy(self.job_state.get_for_analysis_id(uuid))
        for job_id in jobs:
            await self.job_state.remove(job_id)

    def remove_results(self, uuid):
        del self.results[uuid]

    def set_results(self, uuid, details, results, job_id):
        """
        set results: create or update
        """
        self.results[uuid] = (details, results, job_id)

    def get_results(self, uuid):
        return self.results[uuid]

    def get_all_results(self):
        return self.results.items()

    def __getitem__(self, uuid):
        return self.analyses[uuid]

    def serialize(self, uuid):
        return self[uuid]

    def serialize_all(self):
        return [
            self.serialize(uuid)
            for uuid in self.analyses
        ]


class CompoundAnalysisState:
    def __init__(self, analysis_state: AnalysisState):
        self.analysis_state = analysis_state
        self.analyses = {}

    def create_or_update(self, uuid, main_type, dataset_id, analyses):
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

    def remove(self, uuid):
        del self.analyses[uuid]

    def __getitem__(self, uuid):
        return self.analyses[uuid]

    def filter(self, predicate):
        return [
            ca
            for ca in self.analyses.values()
            if predicate(ca)
        ]

    def serialize(self, uuid):
        return self[uuid]

    def serialize_all(self):
        return [
            self.serialize(uuid)
            for uuid in self.analyses
        ]


class DatasetState:
    def __init__(self, executor_state: ExecutorState, analysis_state: AnalysisState,
                 compound_analysis_state: CompoundAnalysisState):
        self.datasets = {}
        self.dataset_to_id = {}
        self.executor_state = executor_state
        self.analysis_state = analysis_state
        self.compound_analysis_state = compound_analysis_state

    def register(self, uuid, dataset, params, converted):
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
        self.jobs = {}
        self.executor_state = executor_state
        self.jobs_for_dataset = defaultdict(lambda: set())
        self.jobs_for_analyses = defaultdict(lambda: set())

    def register(self, job_id, analysis_id, dataset_id):
        assert job_id not in self.jobs
        self.jobs[job_id] = {
            "id": job_id,
            "analysis": analysis_id,
            "dataset": dataset_id,
        }
        self.jobs_for_dataset[dataset_id].add(job_id)
        self.jobs_for_analyses[analysis_id].add(job_id)
        return self

    async def remove(self, uuid):
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

    def get_for_dataset_id(self, dataset_id):
        return self.jobs_for_dataset[dataset_id]

    def get_for_analysis_id(self, analysis_id):
        return self.jobs_for_analyses[analysis_id]

    def __getitem__(self, uuid):
        return self.jobs[uuid]

    def is_cancelled(self, uuid):
        return uuid not in self.jobs

    def serialize(self, job_id):
        job = self[job_id]
        return {
            "id": job["id"],
            "analysis": job["analysis"],
        }

    def serialize_all(self):
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

        self.dataset_for_job = {}
        self.local_directory = "dask-worker-space"

    def get_local_cores(self, default=2):
        cores = psutil.cpu_count(logical=False)
        if cores is None:
            cores = default
        return cores

    def set_local_directory(self, local_directory):
        if local_directory is not None:
            self.local_directory = local_directory

    def get_local_directory(self):
        return self.local_directory

    def get_config(self):
        detected_devices = devices.detect()
        return {
            "version": libertem.__version__,
            "resultFileFormats": ResultFormatRegistry.get_available_formats(),
            "revision": libertem.revision,
            "localCores": self.get_local_cores(),
            "devices": detected_devices,
            "cwd": os.getcwd(),
            # '/' works on Windows, too.
            "separator": '/'
        }
