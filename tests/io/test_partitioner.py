import numpy as np
import pytest

from libertem.common.shape import Shape
from libertem.io.dataset.base.dataset import PartitioningConstraints
from libertem.io.partitioner import (
    PartitionGenerator, get_stop_for_roi, ConstantPartitioner,
)


def test_stop_for_roi_1():
    roi = np.zeros((1000,), dtype=bool)
    roi[::10] = True
    stop = get_stop_for_roi(
        0,
        10,
        roi
    )
    assert stop == 91
    assert np.count_nonzero(roi[:stop]) == 10


def test_stop_for_roi_2():
    roi = np.zeros((1000,), dtype=bool)
    roi[::10] = True
    stop = get_stop_for_roi(
        0,
        1,
        roi
    )
    assert stop == 1
    assert np.count_nonzero(roi[:stop]) == 1


def test_stop_for_empty_roi():
    roi = np.zeros((1000,), dtype=bool)
    stop = get_stop_for_roi(
        0,
        1,
        roi
    )
    assert stop == 0
    assert np.count_nonzero(roi[:stop]) == 0


def test_get_partition_slice_1():
    ds_shape = Shape((100, 16, 16), sig_dims=2)
    p = ConstantPartitioner(
        dataset_shape=ds_shape,
        roi=None,
        partition_size=42,
    )
    start, stop = p.next_partition_slice()
    assert stop - start == 42
    assert stop > start


def test_get_partition_slice_with_roi():
    ds_shape = Shape((100, 16, 16), sig_dims=2)
    roi = np.zeros((100,), dtype=bool)
    roi[::2] = True
    p = ConstantPartitioner(
        dataset_shape=ds_shape,
        roi=roi,
        partition_size=42,
    )
    start, stop = p.next_partition_slice()
    assert stop > start
    assert stop - start == 83
    assert np.count_nonzero(roi[start:stop]) == 42


def test_get_partition_slice_with_zero_roi():
    ds_shape = Shape((100, 16, 16), sig_dims=2)
    roi = np.zeros((100,), dtype=bool)
    p = ConstantPartitioner(
        dataset_shape=ds_shape,
        roi=roi,
        partition_size=42,
    )
    assert p.is_done()

    with pytest.raises(AssertionError):
        p.next_partition_slice()


def test_get_partition_slice_with_zero_roi_at_the_end():
    ds_shape = Shape((100, 16, 16), sig_dims=2)
    roi = np.zeros((100,), dtype=bool)
    roi[:50] = True
    p = ConstantPartitioner(
        dataset_shape=ds_shape,
        roi=roi,
        partition_size=50,
    )
    start, stop = p.next_partition_slice()
    assert stop > start
    assert start == 0
    assert stop == 50
    assert np.count_nonzero(roi[start:stop]) == 50

    assert p.is_done()

    # "mock" the dataset, as it shouldn't be needed to come to the conclusion
    # that the iterator is done
    pg = PartitionGenerator(
        partitioner=p,
        dataset=None,
    )

    with pytest.raises(StopIteration):
        next(pg)


def test_get_partition_slice_with_constraints():
    ds_shape = Shape((100, 16, 16), sig_dims=2)
    p = ConstantPartitioner(
        dataset_shape=ds_shape,
        roi=None,
        partition_size=42,
    )
    constraints = PartitioningConstraints(
        base_step_size=73,
        bytes_per_nav=1024*1024*8,
    )
    start, stop = p.next_partition_slice(
        constraints=constraints,
    )
    assert stop - start == 73
    assert stop > start

    start, stop = p.next_partition_slice(
        constraints=constraints,
    )
    assert stop - start == 73
    assert stop > start
