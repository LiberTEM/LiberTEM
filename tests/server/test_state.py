import uuid
import pytest

from libertem.executor.base import AsyncAdapter
from libertem.executor.dask import DaskJobExecutor
from libertem.web.state import JobState, ExecutorState, SharedState


pytestmark = []


@pytest.mark.asyncio
async def test_job_state_remove(async_executor):
    executor_state = ExecutorState()
    await executor_state.set_executor(async_executor, {})
    job_state = JobState(executor_state=executor_state)

    job_id = str(uuid.uuid4())
    analysis_id = str(uuid.uuid4())
    dataset_id = str(uuid.uuid4())

    job_state.register(job_id, analysis_id, dataset_id)

    assert job_state[job_id] == {
        "id": job_id,
        "analysis": analysis_id,
        "dataset": dataset_id,
    }
    assert job_state.get_for_dataset_id(dataset_id) == {job_id}
    assert job_state.get_for_analysis_id(analysis_id) == {job_id}

    await job_state.remove(job_id)

    print(job_state.jobs)

    with pytest.raises(KeyError):
        job_state[job_id]

    assert job_state.get_for_dataset_id(dataset_id) == set()
    assert job_state.get_for_analysis_id(analysis_id) == set()
    assert len(job_state.jobs) == 0


@pytest.mark.web_api
def test_preload_executor(tmpdir_factory):
    workdir = tmpdir_factory.mktemp('preload_workdir')
    state = SharedState()
    state.set_local_directory(workdir)
    state.set_preload(())
    state.add_executor(
        {
            'cpus': 2,
            'cudas': 0,
        },
    )
    assert isinstance(state.executor_state.executor, AsyncAdapter)
    sync_exec = state.executor_state.executor.ensure_sync()
    assert isinstance(sync_exec, DaskJobExecutor)
    resources = sync_exec.get_resource_details()
    assert resources[0]['cpu'] == 2
    assert resources[0]['cuda'] == 0
    assert resources[0]['service'] == 1

    def test_fn():
        return 42

    assert 42 == sync_exec.run_function(test_fn)
