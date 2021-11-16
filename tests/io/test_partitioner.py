import numpy as np
from libertem.common.shape import Shape

from libertem.io.partitioner import (
    get_stop_for_roi, ConstantPartitioner,
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


def test_get_partition_slice_1():
    ds_shape = Shape((100, 16, 16), sig_dims=2)
    p = ConstantPartitioner(
        dataset_shape=ds_shape,
        target_feedback_rate_hz=10,
        roi=None,
        partition_size=42,
    )
    start, stop = p.get_partition_slice("worker_1")
    assert stop - start == 42


def test_get_partition_slice_with_roi():
    ds_shape = Shape((100, 16, 16), sig_dims=2)
    roi = np.zeros((100,), dtype=bool)
    roi[::2] = True
    p = ConstantPartitioner(
        dataset_shape=ds_shape,
        target_feedback_rate_hz=10,
        roi=roi,
        partition_size=42,
    )
    start, stop = p.get_partition_slice("worker_1")
    assert stop - start == 83
    assert np.count_nonzero(roi[start:stop]) == 42
