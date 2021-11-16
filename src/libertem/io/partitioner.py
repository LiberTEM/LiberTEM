from typing import DefaultDict, List, NamedTuple, Optional, Tuple
from collections import defaultdict

import numpy as np

from libertem.common import Shape
from libertem.common.math import prod

# Constraints (known at initialization):
# UDF: upper limit
# executor / worker:
# - upper constraint by available memory (hard constraint for GPUs)
# - data locality? (-> maybe start this later?)
# DataSet:
# - "base shape"
# - total shape
# user: target feedback rate

# Runtime constraints:
# - different processing rates (GPU vs CPU on the same node, and much different on different nodes)


def get_stop_for_roi(
    start: int,
    size: int,
    roi: np.ndarray,
    step: int = 1,  # TODO: step is the dataset slicing constraint
):
    """
    Get the stop position for a slice in flat dataset coordinates
    that has `size` non-zero entries in `roi`, starting at `start`
    """
    counter = 0
    for idx, roi_entry in enumerate(roi[start:]):
        if roi_entry:
            counter += 1
            stop = start + idx + 1
        if counter >= size:
            break
    return stop


class TaskStats(NamedTuple):
    worker_id: str
    task_size_nav: int
    duration_seconds: float


class Partitioner:
    def __init__(
        self,
        dataset_shape: Shape,
        target_feedback_rate_hz,
        roi: Optional[np.ndarray] = None,
    ):
        self._target_rate = target_feedback_rate_hz
        self._dataset_shape = dataset_shape
        self._total_size = dataset_shape.nav.size
        if roi is not None:
            if prod(roi.shape) != self._total_size:
                raise ValueError(
                    f"roi should match dataset nav shape; "
                    f"is {roi.shape} should be {self._total_size}"
                )
            roi = roi.reshape((-1,))
        self._roi = roi

        # dynamic properties:
        # collected stats
        self._stats_per_worker: DefaultDict[str, List[TaskStats]] = defaultdict(lambda: [])

        # current "position"
        # we generate partitions from the beginning to the end, this is the index
        # where the next partition will start (in flat navigation coords, not taking
        # the `roi` into account):
        self._seek_pos = 0

    def add_task_stats(self, stats: TaskStats):
        self._stats_per_worker[stats.worker_id].append(stats)

    def is_done(self):
        return self._seek_pos >= self._total_size

    def get_partition_slice(self, worker_id: str) -> Tuple[int, int]:
        """
        Get next partition size for task to be scheduled on `worker_id`

        NOTE: not thread safe!
        """
        size = self.calc_partition_size(worker_id)

        start = self._seek_pos

        if self._roi is None:
            stop = start + size
        else:
            stop = get_stop_for_roi(start, size, self._roi)

        self._seek_pos = stop
        return (start, stop)

    def calc_partition_size(self, worker_id: str) -> int:
        # all_stats = self._stats_per_worker[worker_id]
        return 42  # FIXME: calculate desired partition size


class ConstantPartitioner(Partitioner):
    def __init__(self, partition_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._const_size = partition_size

    def calc_partition_size(self, worker_id: str) -> int:
        return self._const_size
