import numpy as np
import pytest
from libertem.slice import Slice


def test_subslices_simple():
    top_slice = Slice(
        origin=(0, 0, 0, 0),
        shape=(4, 4, 4, 4),
    )
    assert list(top_slice.subslices(shape=(2, 2, 4, 4))) == [
        Slice(origin=(0, 0, 0, 0), shape=(2, 2, 4, 4)),
        Slice(origin=(0, 2, 0, 0), shape=(2, 2, 4, 4)),
        Slice(origin=(2, 0, 0, 0), shape=(2, 2, 4, 4)),
        Slice(origin=(2, 2, 0, 0), shape=(2, 2, 4, 4)),
    ]


def test_subslices_must_divide_evenly():
    top_slice = Slice(
        origin=(0, 0, 0, 0),
        shape=(10, 10, 10, 10),
    )
    with pytest.raises(AssertionError):
        top_slice.subslices(shape=(3, 3, 10, 10))


def test_broadcast_slice():
    s = Slice(origin=(5, 5), shape=(10, 10, 10, 10))
    assert s.origin == (5, 5, 0, 0)


def test_get_slice_1():
    slice_ = Slice(
        origin=(0, 0, 0, 0),
        shape=(4, 4, 4, 4),
    )
    assert slice_.get() == (
        slice(0, 4),
        slice(0, 4),
        slice(0, 4),
        slice(0, 4),
    )


def test_get_slice_2():
    slice_ = Slice(
        origin=(1, 1, 1, 1),
        shape=(1, 1, 2, 2),
    )
    data = np.arange(4 * 4 * 4 * 4).reshape(4, 4, 4, 4)
    assert slice_.get(data).shape == slice_.shape
    assert np.all(slice_.get(data) == np.array([[[
        [85, 86],
        [89, 90],
    ]]]))
