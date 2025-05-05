import asyncio
import uuid
import pytest
import queue
from unittest import mock

from libertem.executor.base import AsyncAdapter
from libertem.executor.dask import DaskJobExecutor
from libertem.web.event_bus import EventBus
from libertem.web.state import JobState, ExecutorState, SharedState


pytestmark = []


@pytest.mark.asyncio
async def test_job_state_remove(async_executor):
    event_bus = EventBus()
    executor_state = ExecutorState(event_bus=event_bus)
    try:
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
    finally:
        executor_state.shutdown()


@pytest.mark.web_api
@pytest.mark.asyncio
async def test_preload_executor(tmpdir_factory):
    workdir = tmpdir_factory.mktemp('preload_workdir')
    event_bus = EventBus()
    executor_state = ExecutorState(event_bus=event_bus)
    sync_exec = None
    try:
        state = SharedState(executor_state=executor_state)
        executor_state.set_local_directory(workdir)
        executor_state.set_preload(())

        import tornado.ioloop
        io_loop = tornado.ioloop.IOLoop.current()
        assert io_loop is not None

        await state.create_and_set_executor(
            {
                'cpus': 2,
                'cudas': 0,
            },
        )
        assert isinstance(state.executor_state.executor, AsyncAdapter)
        sync_exec = state.executor_state.executor.ensure_sync()
        assert isinstance(sync_exec, DaskJobExecutor)
        assert sync_exec.is_local
        resources = sync_exec.get_resource_details()
        assert resources[0]['cpu'] == 2
        assert resources[0]['cuda'] == 0
        assert resources[0]['service'] == 1

        def test_fn():
            return 42

        assert 42 == sync_exec.run_function(test_fn)
    finally:
        if sync_exec is not None:
            sync_exec.close()
        executor_state.shutdown()


@pytest.mark.asyncio
async def test_snooze_last_activity():
    params = {
        "connection": {
            "type": "local",
            "numWorkers": 2,
        }
    }
    event_bus = EventBus()
    pool = AsyncAdapter.make_pool()
    executor_state = ExecutorState(event_bus=event_bus, snooze_timeout=10_000)
    executor = await executor_state.make_executor(params, pool)
    await executor_state.set_executor(executor, params)

    try:
        # each of these activities should reset the last activity timer:
        executor_state.executor.snooze_manager._update_last_activity = mock.Mock()
        _ = await executor_state.get_executor()
        executor_state.executor.snooze_manager._update_last_activity.assert_called()

        executor_state.executor.snooze_manager._update_last_activity = mock.Mock()
        _ = await executor_state.get_context()
        executor_state.executor.snooze_manager._update_last_activity.assert_called()

        executor_state.executor.snooze_manager._update_last_activity = mock.Mock()
        _ = executor_state.get_cluster_params()
        executor_state.executor.snooze_manager._update_last_activity.assert_called()

    finally:
        executor_state.shutdown()


@pytest.mark.asyncio
async def test_get_executor_unsnooze():
    """
    Getting the executor brings it out of a snooze state
    """
    event_bus = EventBus()
    executor_state = ExecutorState(event_bus=event_bus, snooze_timeout=10_000)
    pool = AsyncAdapter.make_pool()
    try:
        params = {
            "connection": {
                "type": "local",
                "numWorkers": 2,
            }
        }
        executor = await executor_state.make_executor(params, pool)
        await executor_state.set_executor(executor, params)

        assert executor_state.executor.snooze_manager is not None
        executor_state.executor.snooze_manager.snooze()
        assert executor_state.executor.snooze_manager.is_snoozing
        # Getting the executor brings it out of snooze
        await executor_state.get_executor()
        assert not executor_state.executor.snooze_manager.is_snoozing
    finally:
        pool.shutdown()
        executor_state.shutdown()


@pytest.mark.asyncio
async def test_snooze_explicit_keep_alive():
    """
    We can't snooze if keep-alive is nonzero
    """
    event_bus = EventBus()
    executor_state = ExecutorState(event_bus=event_bus, snooze_timeout=10_000)
    pool = AsyncAdapter.make_pool()
    try:
        params = {
            "connection": {
                "type": "local",
                "numWorkers": 2,
            }
        }
        executor = await executor_state.make_executor(params, pool)
        await executor_state.set_executor(executor, params)

        snoozer = executor_state.executor.snooze_manager
        # if we are in at least one keep-alive section, we can't snooze:
        with snoozer.in_use():
            assert snoozer.keep_alive > 0
            snoozer.snooze()
            assert not snoozer.is_snoozing

        # keep-alive can nest:
        with snoozer.in_use():
            with snoozer.in_use():
                assert snoozer.keep_alive > 0
                snoozer.snooze()
                assert not snoozer.is_snoozing

        # afterwards, we can snooze again:
        assert snoozer.keep_alive == 0
        snoozer.snooze()
        assert snoozer.is_snoozing
        snoozer.unsnooze()
        assert not snoozer.is_snoozing
        # these two work without raising an exception:
        await executor_state.get_executor()
    finally:
        pool.shutdown()
        executor_state.shutdown()


@pytest.mark.asyncio
async def test_snooze_by_activity(local_cluster_url):
    """
    Test that timer-based snoozing works
    """
    pool = AsyncAdapter.make_pool()
    event_bus = EventBus()
    executor_state = ExecutorState(snooze_timeout=0.01, event_bus=event_bus)
    try:
        params = {
            "connection": {
                "type": "local",
                "numWorkers": 2,
            }
        }
        executor = await executor_state.make_executor(params, pool)
        # we must check very frequently; by default we only check twice a minute
        # to keep activity low:
        await executor_state.set_executor(executor, params)
        snoozer = executor_state.executor.snooze_manager
        snoozer._snooze_check_interval = 0.01

        await asyncio.sleep(0.1)
        # after this sleep, the executor should be snoozed, as we had about ten
        # opportunities to snooze in between:
        assert snoozer.is_snoozing

        # and this should directly unsnooze the executor
        # (we need to change the timeout etc. here, before we trigger the unsnooze,
        # to make sure we don't directly snooze again):
        snoozer._snooze_timeout = 3600.0
        _ = await executor_state.get_executor()
        assert not snoozer.is_snoozing

        # should not raise an exception, there should be a snooze message
        # in here (raises Empty otherwise):
        event_bus.get()
    finally:
        executor_state.shutdown()


@pytest.mark.asyncio
async def test_messages():
    """
    Test that we get snooze messages on the event bus
    """
    pool = AsyncAdapter.make_pool()
    event_bus = EventBus()
    executor_state = ExecutorState(snooze_timeout=0.02, event_bus=event_bus)
    try:
        params = {
            "connection": {
                "type": "local",
                "numWorkers": 2,
            }
        }
        executor = await executor_state.make_executor(params, pool)
        # we must check very frequently; by default we only check twice a minute
        # to keep activity low:
        await executor_state.set_executor(executor, params)

        await asyncio.sleep(0.1)

        # these two work without raising an exception:
        await executor_state.get_executor()
        messages = []
        while True:
            try:
                messages.append(event_bus.get()['messageType'])
            except queue.Empty:
                break
        assert "SNOOZE" in messages
        assert "UNSNOOZE" in messages
        assert "UNSNOOZE_DONE" in messages
    finally:
        executor_state.shutdown()
