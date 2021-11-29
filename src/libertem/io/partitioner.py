import logging
import math
from typing import Iterable, List, NamedTuple, Optional, Tuple

import numpy as np
import numba

from libertem.common import Shape
from libertem.common.math import prod
from libertem.common.slice import Slice
from libertem.io.dataset.base.dataset import DataSet, PartitioningConstraints
from libertem.io.dataset.base.partition import Partition
from libertem.io.dataset.base import TilingScheme

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


def fill_contiguous(
    shape: Iterable[int],
    containing_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """
    Make `shape` C-contiguous in `containing_shape`
    (meaining each dimension is either the last one > 1 from right to left, or
    is equal to the corresponding dimension in `containing_shape`)

    Examples
    --------

    >>> ds_shape = (64, 64, 128, 128)
    >>> fill_contiguous((1, 4, 32, 32), ds_shape)
    (1, 4, 128, 128)
    >>> fill_contiguous((2, 4, 32, 32), ds_shape)
    (2, 64, 128, 128)
    >>> fill_contiguous((2, 64, 32, 32), ds_shape)
    (2, 64, 128, 128)
    >>> fill_contiguous((64, 1, 32, 32), ds_shape)
    (64, 64, 128, 128)
    >>> ds_shape_5d = (16, 64, 64, 128, 128)
    >>> fill_contiguous((1, 1, 2, 32, 32), ds_shape_5d)
    (1, 1, 2, 128, 128)
    >>> fill_contiguous((1, 2, 1, 32, 32), ds_shape_5d)
    (1, 2, 64, 128, 128)
    >>> fill_contiguous((2, 1, 1, 32, 32), ds_shape_5d)
    (2, 64, 64, 128, 128)
    >>> fill_contiguous((1, 1, 1), (16, 16, 16))
    (1, 1, 1)
    """
    shape = tuple(shape)
    non_ones = [x == 1 for x in shape]
    if False not in non_ones:
        return shape
    first_non_one = non_ones.index(False)
    shape_left = shape[:first_non_one + 1]
    return shape_left + containing_shape[first_non_one + 1:]


class PartTiming(NamedTuple):
    """
    Timing per partition
    """
    total: float


class TaskStats(NamedTuple):
    task_size_nav: int
    part_timing: PartTiming


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

        start = self._seek_pos

        if self._roi is None:
            stop = start + size
        else:
            # eagerly grow partition based on the `roi`
            stop = get_stop_for_roi(start, size, self._roi)

        # align `new_size` to be a multiple of `base_step_size`:
        new_size = stop - start
        if constraints and new_size % constraints.base_step_size != 0:
            num_blocks = math.ceil(new_size / constraints.base_step_size)
            new_size = num_blocks * constraints.base_step_size

        # align such that (`start`, `new_size`) is a contiguous slice into `dataset_shape`:
        log.debug(f"before contig align: start={start}, new_size={new_size}")
        if constraints and constraints.need_contiguous:
            # first, make sure `new_size` is not out of bounds for `dataset_shape`:
            nd_shape = self._dataset_shape.nav
            new_size = min(new_size, prod(nd_shape) - start)

            # now, make contiguous:
            flat_nav_slice = Slice(
                origin=(start,),
                shape=Shape((new_size,), sig_dims=0),
            )
            nav_slice = flat_nav_slice.unravel_nav(nd_shape)
            contig = fill_contiguous(nav_slice.shape, tuple(nd_shape))
            new_size = prod(contig)

            # validate: result is compatible to the nd_shape:
            Slice(
                origin=(start,),
                shape=Shape((new_size,), sig_dims=0),
            ).unravel_nav(nd_shape).flatten_nav(nd_shape)

        stop = start + new_size

        if stop == start:
            stop += 1  # must have at least one frame per partition

        log.debug(f"after alignments: start={start}, new_size={new_size}")

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
    """
    Generate a target partition size by analyzing the performance of the past N
    tasks, including the time used on the main node for merging.
    """
    def __init__(
        self,
        target_feedback_rate_hz: int,
        num_workers: int,
        partition_constraints: PartitioningConstraints,
        tiling_scheme: TilingScheme,
        *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self._target_rate = target_feedback_rate_hz
        self._num_workers = num_workers
        self._partition_constraints = partition_constraints
        self._tiling_scheme = tiling_scheme

    def get_mean_perf(self, last_n: int) -> float:
        """
        Get the mean performance (as nav items per second)
        for the last `last_n` tasks.
        """
        items = self._stats[-last_n:-1]
        # print(items)
        perfs = [
            item.task_size_nav / item.part_timing.total
            for item in items
        ]
        mean = sum(perfs) / len(perfs)
        return mean

    def have_stats(self):
        return len(self._stats) > 1

    def calc_partition_size(self) -> int:
        # - minimum _number_ (hard) of partitions should be
        #   `n_workers*2` in any case -> `max_size ~= ds_shape.nav.size / (n_workers*2)`
        #    - what about the very sparse `roi` case? there, having more
        #      partitions will slow things down instead
        # - `min_size` constrained by merge time

        # performance in nav items per second:
        if self.have_stats():
            mean_perf = self.get_mean_perf(last_n=32)
        else:
            bytes_per_nav = self._partition_constraints.bytes_per_nav
            mean_perf = 64*1024*1024 / bytes_per_nav * self._target_rate

        total_items = self._dataset_shape.nav.size
        max_size = total_items / (2 * self._num_workers)

        # FIXME: include merge timing here!
        min_size = 2 * self._tiling_scheme.depth

        # our per-partition budget in seconds:
        target_time_per_part = 1 / self._target_rate

        size = target_time_per_part * mean_perf
        size = max(min_size, size)
        size = min(max_size, size)
        print(f"calc_partition_size -> {size} (mean_perf={mean_perf}, "
              f"max_size={max_size}, min_size={min_size}, target_time={target_time_per_part})")
        return int(math.ceil(size))


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
