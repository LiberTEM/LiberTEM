import numpy as np
from libertem.common.slice import Slice


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


def test_subslices_non_even_division_1():
    top_slice = Slice(
        origin=(0, 0, 0, 0),
        shape=(5, 1, 1, 1),
    )
    assert list(top_slice.subslices(shape=(2, 1, 1, 1))) == [
        Slice(origin=(0, 0, 0, 0), shape=(2, 1, 1, 1)),
        Slice(origin=(2, 0, 0, 0), shape=(2, 1, 1, 1)),
        Slice(origin=(4, 0, 0, 0), shape=(1, 1, 1, 1)),
    ]


def test_subslices_non_even_division_2():
    top_slice = Slice(
        origin=(0, 0, 0, 0),
        shape=(3, 1, 1, 1),
    )
    assert list(top_slice.subslices(shape=(2, 1, 1, 1))) == [
        Slice(origin=(0, 0, 0, 0), shape=(2, 1, 1, 1)),
        Slice(origin=(2, 0, 0, 0), shape=(1, 1, 1, 1)),
    ]


def test_subslices_non_even_division_3():
    top_slice = Slice(
        origin=(0, 0, 0, 0),
        shape=(3, 3, 1, 1),
    )
    assert list(top_slice.subslices(shape=(2, 2, 1, 1))) == [
        Slice(origin=(0, 0, 0, 0), shape=(2, 2, 1, 1)),
        Slice(origin=(0, 2, 0, 0), shape=(2, 1, 1, 1)),
        Slice(origin=(2, 0, 0, 0), shape=(1, 2, 1, 1)),
        Slice(origin=(2, 2, 0, 0), shape=(1, 1, 1, 1)),
    ]


def test_subslices_non_even_division_4():
    top_slice = Slice(
        origin=(1, 0, 0, 0),
        shape=(1, 10, 3838, 3710),
    )
    list(top_slice.subslices(shape=(1, 2, 128, 128)))


def test_subslices_non_even_division_with_origin_1():
    top_slice = Slice(
        origin=(0, 3, 0, 0),
        shape=(3, 3, 1, 1),
    )
    assert list(top_slice.subslices(shape=(2, 2, 1, 1))) == [
        Slice(origin=(0, 3, 0, 0), shape=(2, 2, 1, 1)),
        Slice(origin=(0, 5, 0, 0), shape=(2, 1, 1, 1)),
        Slice(origin=(2, 3, 0, 0), shape=(1, 2, 1, 1)),
        Slice(origin=(2, 5, 0, 0), shape=(1, 1, 1, 1)),
    ]


def test_subslices_non_even_division_with_origin_2():
    top_slice = Slice(
        origin=(0, 3, 0, 0),
        shape=(3, 3, 3, 3),
    )
    assert list(top_slice.subslices(shape=(2, 2, 2, 2))) == [
        Slice(origin=(0, 3, 0, 0), shape=(2, 2, 2, 2)),
        Slice(origin=(0, 3, 0, 2), shape=(2, 2, 2, 1)),
        Slice(origin=(0, 3, 2, 0), shape=(2, 2, 1, 2)),
        Slice(origin=(0, 3, 2, 2), shape=(2, 2, 1, 1)),

        Slice(origin=(0, 5, 0, 0), shape=(2, 1, 2, 2)),
        Slice(origin=(0, 5, 0, 2), shape=(2, 1, 2, 1)),
        Slice(origin=(0, 5, 2, 0), shape=(2, 1, 1, 2)),
        Slice(origin=(0, 5, 2, 2), shape=(2, 1, 1, 1)),

        Slice(origin=(2, 3, 0, 0), shape=(1, 2, 2, 2)),
        Slice(origin=(2, 3, 0, 2), shape=(1, 2, 2, 1)),
        Slice(origin=(2, 3, 2, 0), shape=(1, 2, 1, 2)),
        Slice(origin=(2, 3, 2, 2), shape=(1, 2, 1, 1)),

        Slice(origin=(2, 5, 0, 0), shape=(1, 1, 2, 2)),
        Slice(origin=(2, 5, 0, 2), shape=(1, 1, 2, 1)),
        Slice(origin=(2, 5, 2, 0), shape=(1, 1, 1, 2)),
        Slice(origin=(2, 5, 2, 2), shape=(1, 1, 1, 1)),
    ]


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


def test_slice_intersect_0():
    s1 = Slice(
        origin=(0, 0, 0, 0),
        shape=(2, 2, 2, 2),
    )
    s2 = Slice(
        origin=(0, 0, 0, 0),
        shape=(1, 1, 1, 1),
    )
    assert s1.intersection_with(s2) == s2


def test_slice_intersect_1():
    s1 = Slice(
        origin=(0, 0, 0, 0),
        shape=(2, 2, 2, 2),
    )
    s2 = Slice(
        origin=(3, 3, 3, 3),
        shape=(1, 1, 1, 1),
    )
    res = s1.intersection_with(s2)
    assert res == Slice(
        origin=(3, 3, 3, 3),
        shape=(0, 0, 0, 0),
    )
    assert res.is_null()


def test_slice_intersect_2():
    s1 = Slice(
        origin=(1, 1, 1, 1),
        shape=(2, 2, 2, 2),
    )
    s2 = Slice(
        origin=(0, 0, 0, 0),
        shape=(1, 1, 1, 1),
    )
    res = s1.intersection_with(s2)
    assert res == Slice(
        origin=(1, 1, 1, 1),
        shape=(0, 0, 0, 0),
    )
    assert res.is_null()


def test_slice_intersect_3():
    s1 = Slice(
        origin=(1, 1, 1, 1),
        shape=(2, 2, 2, 2),
    )
    s2 = Slice(
        origin=(0, 0, 0, 0),
        shape=(4, 4, 4, 4),
    )
    res = s1.intersection_with(s2)
    assert res == s1


def test_subslice_from_offset_length_1():
    s1 = Slice(
        origin=(1, 1, 1, 1),
        shape=(2, 2, 2, 2),
    )
    sub1 = s1.subslice_from_offset(offset=0, length=2)
    assert sub1.origin == (1, 1, 1, 1)
    assert sub1.shape == (1, 2, 2, 2)

    sub2 = s1.subslice_from_offset(offset=0, length=4)
    assert sub2.origin == (1, 1, 1, 1)
    assert sub2.shape == (2, 2, 2, 2)


def test_subslice_from_offset_length_2():
    s1 = Slice(
        origin=(0, 0, 0, 0),
        shape=(2, 2, 2, 2),
    )

    sub1 = s1.subslice_from_offset(offset=0, length=2)
    assert sub1.origin == (0, 0, 0, 0)
    assert sub1.shape == (1, 2, 2, 2)

    sub2 = s1.subslice_from_offset(offset=0, length=4)
    assert sub2.origin == (0, 0, 0, 0)
    assert sub2.shape == (2, 2, 2, 2)


def test_subslice_from_offset_length_3():
    s1 = Slice(
        origin=(0, 0, 0, 0),
        shape=(4, 4, 2, 2),
    )

    # can also create subslice that is smaller than one row:
    sub3 = s1.subslice_from_offset(offset=0, length=1)
    assert sub3.origin == (0, 0, 0, 0)
    assert sub3.shape == (1, 1, 2, 2)

    sub3 = s1.subslice_from_offset(offset=1, length=1)
    assert sub3.origin == (0, 1, 0, 0)
    assert sub3.shape == (1, 1, 2, 2)


def test_shift_1():
    s1 = Slice(
        origin=(1, 1, 0, 0),
        shape=(1, 1, 2, 2),
    )

    s2 = Slice(
        origin=(1, 1, 0, 0),
        shape=(1, 1, 4, 4)
    )

    shifted = s1.shift(s2)

    assert shifted.origin == (0, 0, 0, 0)


def test_shift_2():
    s1 = Slice(
        origin=(2, 2, 0, 0),
        shape=(1, 1, 2, 2),
    )

    s2 = Slice(
        origin=(1, 1, 0, 0),
        shape=(1, 1, 4, 4)
    )

    shifted = s1.shift(s2)
    assert shifted.origin == (1, 1, 0, 0)
