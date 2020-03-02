import os
import typing

import psutil

import libertem
from libertem.analysis.base import AnalysisResultSet
from libertem.io.dataset.base import DataSetException


class ExecutorState:
    def __init__(self):
        self.executor = None

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


AnalysisParameters = typing.Dict


class AnalysisState:
    def __init__(self, executor_state: ExecutorState):
        self.analyses: typing.Dict[str, AnalysisParameters] = {}
        self.results: typing.Dict[str, typing.Tuple[AnalysisResultSet, AnalysisParameters]] = {}

    def create_analysis(self, uuid, dataset_uuid, analysis_type, parameters):
        assert uuid not in self.analyses
        self.analyses[uuid] = {
            "type": analysis_type,
            "dataset": dataset_uuid,
            "params": parameters,
        }

    def get_analysis(self, uuid):
        return self.analyses.get(uuid)


class DatasetState:
    def __init__(self, executor_state: ExecutorState):
        self.datasets = {}
        self.executor_state = executor_state

    def register(self, uuid, dataset, params):
        assert uuid not in self.datasets
        self.datasets[uuid] = {
            "dataset": dataset,
            "params": params,
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
        executor = self.executor_state.get_executor()
        return [
            await self.serialize_dataset(executor, dataset_id)
            for dataset_id in self.datasets.keys()
        ]

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
        ds = self.datasets[uuid]["dataset"]
        jobs_to_remove = [
            job
            for job in self.jobs.values()
            if self.dataset_to_id[self.dataset_for_job[job]] == uuid
        ]
        job_ids = {self.job_to_id[job]
                   for job in jobs_to_remove}
        del self.datasets[uuid]
        del self.dataset_to_id[ds]
        for job_id in job_ids:
            await self.remove_job(job_id)


class JobState:
    def __init__(self, executor_state: ExecutorState, dataset_state: DatasetState):
        self.jobs = {}
        self.job_to_id = {}
        self.executor_state = executor_state
        self.dataset_state = dataset_state

    def register(self, uuid, job, analysis_id, dataset):
        assert uuid not in self.jobs
        self.jobs[uuid] = job
        self.job_to_id[job] = uuid
        self.dataset_for_job[job] = dataset
        return self

    async def remove(self, uuid):
        try:
            job = self.jobs[uuid]
            executor = self.executor_state.get_executor()
            await executor.cancel(uuid)
            del self.jobs[uuid]
            del self.job_to_id[job]
            return True
        except KeyError:
            return False

    def __getitem__(self, uuid):
        return self.jobs[uuid]

    def is_cancelled(self, uuid):
        return uuid not in self.jobs

    def serialize(self, job_id):
        job = self.jobs[job_id]
        dataset = self.dataset_to_id[self.dataset_for_job[job]]
        return {
            "id": job_id,
            "dataset": dataset,
        }

    def serialize_all(self):
        return [
            self.serialize_job(job_id)
            for job_id in self.jobs.keys()
        ]


class SharedState:
    def __init__(self):
        self.executor_state = ExecutorState()
        self.analysis_state = AnalysisState(self.executor_state)
        self.dataset_state = DatasetState(self.executor_state)
        self.job_state = JobState(self.executor_state, self.dataset_state)

        self.dataset_to_id = {}
        self.dataset_for_job = {}
        self.cluster_params = {}
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
        return {
            "version": libertem.__version__,
            "revision": libertem.revision,
            "localCores": self.get_local_cores(),
            "cwd": os.getcwd(),
            # '/' works on Windows, too.
            "separator": '/'
        }

    def get_cluster_params(self):
        return self.cluster_params
