import time
import pytest
import typing
from contextlib import contextmanager

import libertem.api as lt
from libertem.udf.sum import SumUDF
from libertem.common.progress import TQDMProgressReporter, ProgressState
from libertem.common.executor import TaskCommHandler, WorkerQueue, WorkerQueueEmpty
from libertem.io.dataset.memory import MemoryDataSet

import utils

if typing.TYPE_CHECKING:
    from libertem.io.dataset.base.tiling import DataTile


class TrackingTQDM(TQDMProgressReporter):
    def __init__(self):
        super().__init__()
        self._history: list[ProgressState] = []

    def start(self, state):
        self._history.append(state)
        super().start(state)

    def update(self, state):
        self._history.append(state)
        super().update(state)

    def end(self, state):
        self._history.append(state)
        super().end(state)

    def print(self):
        for hist in self._history:
            print(hist)


class InlineProgressTaskCommHandler(TaskCommHandler):
    @contextmanager
    def monitor(self, queue: WorkerQueue):
        """
        Normally this would monitor the queue in the background
        and set up callbacks for progress, but instead just save
        a reference to the queue and wait, i.e. all messages received
        will build up in the queue and not be consumed
        """
        self._progress_queue = queue
        yield


class MemoryDataSetMockComms(MemoryDataSet):
    def get_task_comm_handler(self):
        """
        Save reference to patched comm handler so we can
        access the filled queue after the call to run_udf
        """
        self._comms_handler = InlineProgressTaskCommHandler()
        return self._comms_handler


def drain_queue(queue):
    while True:
        try:
            with queue.get(block=False) as ((topic, msg), _):
                yield topic, msg
        except WorkerQueueEmpty:
            break


class FastTime:
    """
    Time getter which increases by at least
    :code:`increment` seconds each call
    """
    def __init__(self, increment: float = 1.5):
        self._now = 0.
        self._increment = increment

    def __call__(self):
        self._now += self._increment
        return self._now


class WaitEndSumUDF(SumUDF):
    def postprocess(self):
        # Need time to let the message queue be processed
        # before the partition completes, as completion updates
        # happen synchronously on the main node and over-ride
        # further tile-level updates
        time.sleep(0.5)


def test_progress_inline_fasttime(lt_ctx, monkeypatch):
    """
    Tests that a comms failure will still correctly report
    Task completion, and that each partition sends signals
    when it begins and for each tile by patching the get_time
    function to increment quickly
    """
    fast_time = FastTime()
    monkeypatch.setattr('libertem.common.progress.get_time', fast_time)

    data = utils._mk_random(size=(4, 4, 16, 16), dtype='float32')
    ds = MemoryDataSetMockComms(
        data=data,
        num_partitions=2,
        tileshape=(4, 4, 16),
    )
    ds.initialize(lt_ctx)

    reporter = TrackingTQDM()
    udf = SumUDF()
    lt_ctx.run_udf(ds, udf, progress=reporter)

    assert reporter._bar.n == reporter._bar.total

    # These are the states which are generated synchronously
    # through a call to finalize_task() on the main thread
    # This tests the fallback mechanism in case comms fail
    states = reporter._history
    progress_id = states[0].progress_id
    start_progress = ProgressState(0., 16, 0, 0, 2, progress_id)
    assert start_progress in states
    part0_end_progress = ProgressState(8., 16, 1, 0, 2, progress_id)
    assert part0_end_progress in states
    part1_end_progress = ProgressState(16., 16, 2, 0, 2, progress_id)
    assert part1_end_progress in states

    # Get the queue to read the intermediate messages
    queue = ds._comms_handler._progress_queue
    # We expect 7 tile messages per partition as the first is skipped
    expected_order = ['partition_start'] + ['tile_complete'] * 7
    # Two partitions in this dataset
    expected_order = expected_order * 2

    # Drain the queue of messages and check the order
    current_part = None
    for (topic, message), expected_topic in zip(drain_queue(queue), expected_order):
        assert topic == expected_topic
        if topic == 'partition_start':
            current_part = message['ident']
        assert message['ident'] == current_part


class GoSlowSumUDF(WaitEndSumUDF):
    def process_tile(self, tile: 'DataTile'):
        super().process_tile(tile)
        # Sleep on the second tile
        # processed each partition
        try:
            if self._tiles_seen == 1:
                time.sleep(1.5)
        except AttributeError:
            self._tiles_seen = 0
        self._tiles_seen += 1


def _test_progress_slowudf(context: lt.Context):
    data = utils._mk_random(size=(4, 4, 16, 16), dtype='float32')
    ds = context.load(
        'memory',
        data=data,
        num_partitions=2,
        tileshape=(4, 4, 16),
    )

    reporter = TrackingTQDM()
    udf = GoSlowSumUDF()
    context.run_udf(ds, udf, progress=reporter)

    assert reporter._bar.n == reporter._bar.total
    num_part = ds.get_num_partitions()
    # Run start / stop + (part start/stop * num_part)
    min_num_messages = 2 + num_part * 2
    # We should recieve at least one extra message per part with this UDF
    # The only risk is if the postprocess time is not long enough for the
    # tile message to be processed!
    assert len(reporter._history) >= (min_num_messages + num_part)


@pytest.mark.slow
def test_progress_inline_slowudf(lt_ctx):
    _test_progress_slowudf(lt_ctx)


@pytest.mark.slow
def test_progress_concurrent_slowudf(concurrent_ctx):
    _test_progress_slowudf(concurrent_ctx)


@pytest.mark.slow
def test_progress_dask(dask_executor, default_raw):
    reporter = TrackingTQDM()
    udf = GoSlowSumUDF()
    runner_cls = dask_executor.get_udf_runner()
    runner = runner_cls([udf], progress_reporter=reporter)
    runner.run_for_dataset(
        default_raw,
        dask_executor,
        roi=None,
        progress=True
    )

    assert reporter._bar.n == reporter._bar.total
    num_part = default_raw.get_num_partitions()
    # Run start / stop + (part start / stop * num_part)
    min_num_messages = 2 + num_part * 2
    # default_raw doesn't have enough tiles to send intermediate
    # tile messages (depending on the system resources), but we can at least
    # check the comms are working by verifying we had min_num_messages
    assert len(reporter._history) >= min_num_messages


@pytest.mark.slow
def test_progress_pipelined(default_raw):
    with lt.Context.make_with("pipelined") as ctx:
        reporter = TrackingTQDM()
        udf = GoSlowSumUDF()
        ctx.run_udf(default_raw, udf, progress=reporter)

        assert reporter._bar.n == reporter._bar.total
        num_part = default_raw.get_num_partitions()
        # Run start / stop + (part start / stop * num_part)
        min_num_messages = 2 + num_part * 2
        # default_raw doesn't have enough tiles to send intermediate
        # tile messages (depending on the system resources), but we can at least
        # check the comms are working by verifying we had min_num_messages
        assert len(reporter._history) >= min_num_messages
