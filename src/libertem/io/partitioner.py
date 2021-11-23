import logging
import math
from typing import List, NamedTuple, Optional, Tuple

import numpy as np
import numba

from libertem.common import Shape
from libertem.common.math import prod
from libertem.io.dataset.base.dataset import DataSet, PartitioningConstraints
from libertem.io.dataset.base.partition import Partition

log = logging.getLogger(__name__)


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
# - merge time places a lower limit on the time - if merging takes too long, it will slow down
#   the overall result if the partitions are too small (tasks are "too fast")


@numba.njit(cache=True)  # jit: about 40x faster
def get_stop_for_roi(
    start: int,
    size: int,
    roi: np.ndarray,
):
    """
    Get the stop position for a slice in flat dataset coordinates
    that has `size` non-zero entries in `roi`, starting at `start`
    """
    counter = 0
    stop = start
    for idx, roi_entry in enumerate(roi[start:]):
        if roi_entry:
            counter += 1
            stop = start + idx + 1
        if counter >= size:
            break
    return stop


class TaskStats(NamedTuple):
    task_size_nav: int
    duration_seconds: float


class Partitioner:
    def __init__(
        self,
        dataset_shape: Shape,
        roi: Optional[np.ndarray] = None,
    ):
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
        self._stats: List[TaskStats] = []

        # current "position"
        # we generate partitions from the beginning to the end, this is the index
        # where the next partition will start (in flat navigation coords, not taking
        # the `roi` into account):
        self._seek_pos = 0

    def __repr__(self) -> str:
        return f"<Partitioner @ {self._seek_pos}>"

    def add_task_stats(self, stats: TaskStats) -> None:
        print(stats)
        self._stats.append(stats)

    def add_merge_stats(self) -> None:
        """
        If the merge function is limiting our performance, we
        want to lower the effective update rate.
        """
        pass  # FIXME

    def is_done(self) -> bool:
        done_pos = self._seek_pos >= self._total_size
        done_roi = self._roi is not None and np.count_nonzero(self._roi[self._seek_pos:]) == 0
        return done_pos or done_roi

    def next_partition_slice(
        self,
        constraints: Optional[PartitioningConstraints] = None
    ) -> Tuple[int, int]:
        """
        Get next partition size.

        The generated (start, stop) tuples are guaranteed to span
        the dataset from the first index to at least the last index
        that is covered by the roi.

        NOTE: not thread safe! depends on and changes the seek position
        non-atomically
        """
        assert not self.is_done()
        size = self.calc_partition_size()

        if constraints and size % constraints.base_step_size != 0:
            num_blocks = math.ceil(size / constraints.base_step_size)
            size = num_blocks * constraints.base_step_size

        start = self._seek_pos

        if self._roi is None:
            stop = start + size
        else:
            stop = get_stop_for_roi(start, size, self._roi)

        if stop == start:
            stop += 1  # must have at least one frame per partition

        log.debug(f"yielding slice: {start}, {stop}")
        log.debug(f"seeking from {self._seek_pos} to {stop}")
        self._seek_pos = stop

        return (start, stop)

    def calc_partition_size(self) -> int:
        raise NotImplementedError()


class ConstantPartitioner(Partitioner):
    def __init__(self, partition_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._const_size = partition_size

    def calc_partition_size(self) -> int:
        return self._const_size


class AdaptivePartitioner(Partitioner):
    def __init__(self, target_feedback_rate_hz: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_rate = target_feedback_rate_hz

    def calc_partition_size(self) -> int:
        # - minimum _number_ (hard) of partitions should be
        #   `n_workers*2` in any case -> `max_size ~= ds_shape.nav.size / (n_workers*2)`
        # - `min_size` constraint by merge time
        return 42  # FIXME: implement this!


class PartitionGenerator:
    """
    Generate partitions from the :class:`Partitioner` and the `DataSet`.

    There can be multiple :class:`PartitionGenerator` instances, sharing
    the same :class`Partitioner` instance.
    """
    def __init__(
        self,
        partitioner: Partitioner,
        dataset: DataSet,
        part_constraints: Optional[PartitioningConstraints] = None,
    ):
        self.partitioner = partitioner
        self.dataset = dataset
        self.part_constraints = part_constraints

    def __iter__(self) -> "PartitionGenerator":
        return self

    def __next__(self) -> Partition:
        if self.partitioner.is_done():
            raise StopIteration()
        start, stop = self.partitioner.next_partition_slice(
            constraints=self.part_constraints,
        )
        assert stop > start
        return self.dataset.get_partition_for_slice(start, stop)

    def add_task_stats(self, stats: TaskStats) -> None:
        self.partitioner.add_task_stats(stats)
