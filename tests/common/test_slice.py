import pytest
import numpy as np
from libertem.common import Slice, Shape, SliceUsageError


def test_subslices_simple():
    top_slice = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((4, 4, 4, 4), sig_dims=2),
    )
    assert list(top_slice.subslices(shape=(2, 2, 4, 4))) == [
        Slice(origin=(0, 0, 0, 0), shape=Shape((2, 2, 4, 4), sig_dims=2)),
        Slice(origin=(0, 2, 0, 0), shape=Shape((2, 2, 4, 4), sig_dims=2)),
        Slice(origin=(2, 0, 0, 0), shape=Shape((2, 2, 4, 4), sig_dims=2)),
        Slice(origin=(2, 2, 0, 0), shape=Shape((2, 2, 4, 4), sig_dims=2)),
    ]


def test_subslices_non_even_division_1():
    top_slice = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((5, 1, 1, 1), sig_dims=2),
    )
    assert list(top_slice.subslices(shape=(2, 1, 1, 1))) == [
        Slice(origin=(0, 0, 0, 0), shape=Shape((2, 1, 1, 1), sig_dims=2)),
        Slice(origin=(2, 0, 0, 0), shape=Shape((2, 1, 1, 1), sig_dims=2)),
        Slice(origin=(4, 0, 0, 0), shape=Shape((1, 1, 1, 1), sig_dims=2)),
    ]


def test_subslices_non_even_division_2():
    top_slice = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((3, 1, 1, 1), sig_dims=2),
    )
    assert list(top_slice.subslices(shape=(2, 1, 1, 1))) == [
        Slice(origin=(0, 0, 0, 0), shape=Shape((2, 1, 1, 1), sig_dims=2)),
        Slice(origin=(2, 0, 0, 0), shape=Shape((1, 1, 1, 1), sig_dims=2)),
    ]


def test_subslices_non_even_division_3():
    top_slice = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((3, 3, 1, 1), sig_dims=2),
    )
    assert list(top_slice.subslices(shape=(2, 2, 1, 1))) == [
        Slice(origin=(0, 0, 0, 0), shape=Shape((2, 2, 1, 1), sig_dims=2)),
        Slice(origin=(0, 2, 0, 0), shape=Shape((2, 1, 1, 1), sig_dims=2)),
        Slice(origin=(2, 0, 0, 0), shape=Shape((1, 2, 1, 1), sig_dims=2)),
        Slice(origin=(2, 2, 0, 0), shape=Shape((1, 1, 1, 1), sig_dims=2)),
    ]


def test_subslices_non_even_division_4():
    top_slice = Slice(
        origin=(1, 0, 0, 0),
        shape=Shape((1, 10, 3838, 3710), sig_dims=2),
    )
    list(top_slice.subslices(shape=(1, 2, 128, 128)))


def test_subslices_non_even_division_with_origin_1():
    top_slice = Slice(
        origin=(0, 3, 0, 0),
        shape=Shape((3, 3, 1, 1), sig_dims=2),
    )
    assert list(top_slice.subslices(shape=(2, 2, 1, 1))) == [
        Slice(origin=(0, 3, 0, 0), shape=Shape((2, 2, 1, 1), sig_dims=2)),
        Slice(origin=(0, 5, 0, 0), shape=Shape((2, 1, 1, 1), sig_dims=2)),
        Slice(origin=(2, 3, 0, 0), shape=Shape((1, 2, 1, 1), sig_dims=2)),
        Slice(origin=(2, 5, 0, 0), shape=Shape((1, 1, 1, 1), sig_dims=2)),
    ]


def test_subslices_non_even_division_with_origin_2():
    top_slice = Slice(
        origin=(0, 3, 0, 0),
        shape=Shape((3, 3, 3, 3), sig_dims=2),
    )
    assert list(top_slice.subslices(shape=(2, 2, 2, 2))) == [
        Slice(origin=(0, 3, 0, 0), shape=Shape((2, 2, 2, 2), sig_dims=2)),
        Slice(origin=(0, 3, 0, 2), shape=Shape((2, 2, 2, 1), sig_dims=2)),
        Slice(origin=(0, 3, 2, 0), shape=Shape((2, 2, 1, 2), sig_dims=2)),
        Slice(origin=(0, 3, 2, 2), shape=Shape((2, 2, 1, 1), sig_dims=2)),

        Slice(origin=(0, 5, 0, 0), shape=Shape((2, 1, 2, 2), sig_dims=2)),
        Slice(origin=(0, 5, 0, 2), shape=Shape((2, 1, 2, 1), sig_dims=2)),
        Slice(origin=(0, 5, 2, 0), shape=Shape((2, 1, 1, 2), sig_dims=2)),
        Slice(origin=(0, 5, 2, 2), shape=Shape((2, 1, 1, 1), sig_dims=2)),

        Slice(origin=(2, 3, 0, 0), shape=Shape((1, 2, 2, 2), sig_dims=2)),
        Slice(origin=(2, 3, 0, 2), shape=Shape((1, 2, 2, 1), sig_dims=2)),
        Slice(origin=(2, 3, 2, 0), shape=Shape((1, 2, 1, 2), sig_dims=2)),
        Slice(origin=(2, 3, 2, 2), shape=Shape((1, 2, 1, 1), sig_dims=2)),

        Slice(origin=(2, 5, 0, 0), shape=Shape((1, 1, 2, 2), sig_dims=2)),
        Slice(origin=(2, 5, 0, 2), shape=Shape((1, 1, 2, 1), sig_dims=2)),
        Slice(origin=(2, 5, 2, 0), shape=Shape((1, 1, 1, 2), sig_dims=2)),
        Slice(origin=(2, 5, 2, 2), shape=Shape((1, 1, 1, 1), sig_dims=2)),
    ]


def test_get_slice_1():
    slice_ = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((4, 4, 4, 4), sig_dims=2),
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
        shape=Shape((1, 1, 2, 2), sig_dims=2),
    )
    data = np.arange(4 * 4 * 4 * 4).reshape(4, 4, 4, 4)
    assert slice_.get(data).shape == tuple(slice_.shape)
    assert np.all(slice_.get(data) == np.array([[[
        [85, 86],
        [89, 90],
    ]]]))


def test_get_slice_stack_signal_only():
    slice_ = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((1, 1, 1, 1), sig_dims=2)
    )

    data = np.arange(4 * 4 * 4 * 4).reshape(4, 4, 4, 4)
    assert slice_.get(data, sig_only=True).shape[2:4] == tuple(slice_.shape.sig)
    assert np.all(slice_.get(data, sig_only=True) == data[..., 0:1, 0:1])


def test_get_slice_stack_nav_only():
    slice_ = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((1, 1, 1, 1), sig_dims=2)
    )

    data = np.arange(4 * 4 * 4 * 4).reshape(4, 4, 4, 4)
    assert slice_.get(data, nav_only=True).shape[0:2] == tuple(slice_.shape.nav)
    assert np.all(slice_.get(data, nav_only=True) == data[0:1, 0:1])


def test_slice_intersect_0():
    s1 = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((2, 2, 2, 2), sig_dims=2),
    )
    s2 = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((1, 1, 1, 1), sig_dims=2),
    )
    assert s1.intersection_with(s2) == s2


def test_slice_intersect_1():
    s1 = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((2, 2, 2, 2), sig_dims=2),
    )
    s2 = Slice(
        origin=(3, 3, 3, 3),
        shape=Shape((1, 1, 1, 1), sig_dims=2),
    )
    res = s1.intersection_with(s2)
    assert res == Slice(
        origin=(3, 3, 3, 3),
        shape=Shape((0, 0, 0, 0), sig_dims=2),
    )
    assert res.is_null()


def test_slice_intersect_2():
    s1 = Slice(
        origin=(1, 1, 1, 1),
        shape=Shape((2, 2, 2, 2), sig_dims=2),
    )
    s2 = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((1, 1, 1, 1), sig_dims=2),
    )
    res = s1.intersection_with(s2)
    assert res == Slice(
        origin=(1, 1, 1, 1),
        shape=Shape((0, 0, 0, 0), sig_dims=2),
    )
    assert res.is_null()


def test_slice_intersect_3():
    s1 = Slice(
        origin=(1, 1, 1, 1),
        shape=Shape((2, 2, 2, 2), sig_dims=2)
    )
    s2 = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((4, 4, 4, 4), sig_dims=2)
    )
    res = s1.intersection_with(s2)
    assert res == s1


def test_shift_1():
    s1 = Slice(
        origin=(1, 1, 0, 0),
        shape=Shape((1, 1, 2, 2), sig_dims=2)
    )

    s2 = Slice(
        origin=(1, 1, 0, 0),
        shape=Shape((1, 1, 4, 4), sig_dims=2)
    )

    shifted = s1.shift(s2)

    assert shifted.origin == (0, 0, 0, 0)


def test_shift_2():
    s1 = Slice(
        origin=(2, 2, 0, 0),
        shape=Shape((1, 1, 2, 2), sig_dims=2)
    )

    s2 = Slice(
        origin=(1, 1, 0, 0),
        shape=Shape((1, 1, 4, 4), sig_dims=2)
    )

    shifted = s1.shift(s2)
    assert shifted.origin == (1, 1, 0, 0)


def test_get_signal_only():
    s = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((1, 1, 1, 1), sig_dims=2)
    )

    assert s.get(sig_only=True) == (
        slice(0, 1),
        slice(0, 1),
    )


def test_get():
    s = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((1, 1, 1, 1), sig_dims=2)
    )

    assert s.get() == (
        slice(0, 1),
        slice(0, 1),
        slice(0, 1),
        slice(0, 1),
    )


def test_flatten_nav():
    s = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((1, 1, 1, 1), sig_dims=2)
    )
    sflat = Slice(
        origin=(0, 0, 0),
        shape=Shape((1, 1, 1), sig_dims=2)
    )
    assert s.flatten_nav((1, 1, 1, 1)) == sflat


def test_flatten_nav_2():
    s = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((2, 16, 16, 16), sig_dims=2)
    )
    sflat = Slice(
        origin=(0, 0, 0),
        shape=Shape((32, 16, 16), sig_dims=2)
    )
    assert s.flatten_nav((16, 16, 16, 16)) == sflat


def test_slice_to_sig():
    s = Slice(
        origin=(0, 0, 8, 8),
        shape=Shape((2, 16, 16, 16), sig_dims=2)
    )
    assert s.sig.origin == (8, 8)
    assert s.sig.shape.to_tuple() == (16, 16)
    assert s.sig.shape.sig.dims == 2


def test_slice_to_nav():
    s = Slice(
        origin=(3, 3, 8, 8),
        shape=Shape((2, 16, 16, 16), sig_dims=2)
    )
    assert s.nav.origin == (3, 3)
    assert s.nav.shape.to_tuple() == (2, 16)
    assert s.nav.shape.sig.dims == 0


def test_from_shape():
    s = Slice.from_shape(
        (1, 16, 16),
        sig_dims=2
    )
    assert s == Slice(
        origin=(0, 0, 0),
        shape=Shape((1, 16, 16), sig_dims=2),
    )


def test_shift_offset_consistency():
    offset1 = (0, -3, 5)
    offset2 = (1, 7, -19)
    total_offset = tuple(o1 + o2 for (o1, o2) in zip(offset1, offset2))
    s1 = Slice.from_shape(
        (1, 16, 16),
        sig_dims=2
    )
    s2 = s1.shift_by(offset1)
    s3 = s2.shift_by(offset2)
    assert s3.origin == total_offset
    assert s3.shape == s1.shape


def test_slice_raises_not_shape():
    with pytest.raises(SliceUsageError):
        Slice(
            origin=(1, 1, 1, 1),
            shape=(2, 2, 2, 2),
        )


def test_slice_raises_mismatching():
    with pytest.raises(SliceUsageError):
        Slice(
            origin=(1, 1, 1),
            shape=Shape((2, 2, 2, 2), sig_dims=2),
        )


def test_intersections_raises_mismatching():
    s1 = Slice(
        origin=(1, 1, 1, 1),
        shape=Shape((2, 2, 2, 2), sig_dims=2),
    )
    s2 = Slice(
        origin=(1, 1, 1),
        shape=Shape((2, 2, 2), sig_dims=2),
    )
    s3 = Slice(
        origin=(1, 1, 1, 1),
        shape=Shape((2, 2, 2, 2), sig_dims=1),
    )
    with pytest.raises(SliceUsageError):
        s1.intersection_with(s2)
    with pytest.raises(SliceUsageError):
        s1.intersection_with(s3)


def test_shifts_raises_mismatching():
    s1 = Slice(
        origin=(1, 1, 1, 1),
        shape=Shape((2, 2, 2, 2), sig_dims=2),
    )
    s2 = Slice(
        origin=(1, 1, 1),
        shape=Shape((2, 2, 2), sig_dims=2),
    )
    with pytest.raises(SliceUsageError):
        s1.shift(s2)
    with pytest.raises(SliceUsageError):
        s1.shift_by(s2.origin)
