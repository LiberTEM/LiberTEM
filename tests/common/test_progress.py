import time
import pytest
import sys
import typing

import libertem.api as lt
from libertem.udf.sum import SumUDF
from libertem.udf.base import UDFRunner
from libertem.common.progress import TQDMProgressReporter, ProgressState

import utils

if typing.TYPE_CHECKING:
    from libertem.io.dataset.base.tiling import DataTile


class TrackingTQDM(TQDMProgressReporter):
    def __init__(self):
        super().__init__()
        self._history: typing.List[ProgressState] = []

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
        print(self._now)
        return self._now


class WaitEndSumUDF(SumUDF):
    def postprocess(self):
        # Need time to let the message queue be processed
        # before the partition completes, as completion updates
        # happen synchronously on the main node and over-ride
        # further tile-level updates
        time.sleep(0.5)


def test_progress_inline_fasttime(lt_ctx, monkeypatch):
    fast_time = FastTime()
    monkeypatch.setattr('libertem.common.progress.get_time', fast_time)

    data = utils._mk_random(size=(4, 4, 16, 16), dtype='float32')
    ds = lt_ctx.load(
        'memory',
        data=data,
        num_partitions=2,
        tileshape=(4, 4, 16),
    )

    # 1 setup message
    # 1 message per partition start
    # 8 tiles per partition
    # first tile is never signalled
    # final few tile messages might be skipped by partition end message
    # 1 partition end message
    # 1 teardown message

    reporter = TrackingTQDM()
    udf = SumUDF()
    runner = UDFRunner([udf], progress_reporter=reporter)
    runner.run_for_dataset(
        ds,
        lt_ctx.executor,
        roi=None,
        progress=True
    )

    assert reporter._bar.n == reporter._bar.total

    states = reporter._history
    # Check we received some specific messages
    start_progress = ProgressState(0., 16, 0, 0, 2)
    assert start_progress in states
    part0_start_progress = ProgressState(0., 16, 0, 1, 2)
    assert part0_start_progress in states
    part0_end_progress = ProgressState(8., 16, 1, 0, 2)
    assert part0_end_progress in states
    part1_start_progress = ProgressState(8., 16, 1, 1, 2)
    assert part1_start_progress in states
    part1_end_progress = ProgressState(16., 16, 2, 0, 2)
    assert part1_end_progress in states

    # Becase every tile is 'slow' we can expect
    # to recieve some mid-partition updates
    part0_mid_progress = ProgressState(4., 16, 0, 1, 2)
    assert part0_mid_progress in states
    part1_mid_progress = ProgressState(12., 16, 1, 1, 2)
    assert part1_mid_progress in states

    # Check for errant values
    assert {s.num_part_in_progress for s in states} == {0, 1}
    assert {s.num_part_complete for s in states} == {0, 1, 2}
    assert {s.num_part_total for s in states} == {2}
    assert {s.num_frames_total for s in states} == {16}
    # Possible updates excludes 1 and 9 because first tiles are never reported
    possible_frames_complete = {float(r) for r in range(16 + 1)}.difference({1., 9.})
    assert {s.num_frames_complete for s in states}.issubset(possible_frames_complete)


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
    runner = UDFRunner([udf], progress_reporter=reporter)
    runner.run_for_dataset(
        ds,
        context.executor,
        roi=None,
        progress=True
    )

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
@pytest.mark.skipif(sys.version_info < (3, 7),
                    reason="Python3.6 Dask has no comms")
def test_progress_dask(dask_executor, default_raw):
    reporter = TrackingTQDM()
    udf = GoSlowSumUDF()
    runner = UDFRunner([udf], progress_reporter=reporter)
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
        runner = UDFRunner([udf], progress_reporter=reporter)
        runner.run_for_dataset(
            default_raw,
            ctx.executor,
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
