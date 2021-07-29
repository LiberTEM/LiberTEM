import uuid
import pytest

from libertem.web.state import JobState, ExecutorState


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
