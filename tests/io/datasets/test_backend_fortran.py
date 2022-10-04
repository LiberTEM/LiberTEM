import itertools

import pytest

from libertem.io.dataset.base.backend_fortran import FortranReader


@pytest.mark.parametrize(
    "length, depth, result",
    [
        (6, 3, ((0, 3), (3, 6))),
        (5, 2, ((0, 2), (2, 4), (4, 5))),
        (3, 1, ((0, 1), (1, 2), (2, 3))),
    ],
)
def test_gen_slices_for_depth(length, depth, result):
    slices = tuple(FortranReader._gen_slices_for_depth(length, depth))
    assert slices == result
    ranges = tuple(range(*s) for s in slices)
    assert all(0 < len(r) <= depth for r in ranges)
    assert tuple(itertools.chain(*ranges)) == tuple(range(length))


@pytest.mark.parametrize(
    "obj, res",
    [
        (slice(5), 5),
        (slice(10, 16), 6),
        (3, 1)
    ],
)
def test_length_slice(obj, res):
    assert FortranReader._length_slice(obj) == res


@pytest.mark.parametrize(
    "obj",
    [
        slice(5, 20, 3),
        slice(20, 5),
        (5, 6, 7),
        None
    ],
)
def test_length_slice_invalid(obj):
    with pytest.raises(AssertionError):
        FortranReader._length_slice(obj)


@pytest.mark.parametrize(
    "sl, target, result",
    [
        (slice(7), 3, (slice(0, 3), slice(3, 6), slice(6, 7))),
        (slice(6, 10), 2, (slice(6, 8), slice(8, 10))),
        (5, 2, (slice(5, 6),)),
    ],
)
def test_split_slice(sl, target, result):
    slices = tuple(FortranReader._split_slice(sl, target))
    assert slices == result
    assert all(FortranReader._length_slice(sl_) <= target for sl_ in slices)


@pytest.mark.parametrize(
    "seq, result",
    [
        ((0, (1, 2, 3), 4), (0, 1, 2, 3, 4)),
        ((None, [1, 2, 3], slice(5, 7)), (None, 1, 2, 3, slice(5, 7))),
        ((0, 1, 2, 3, 4), (0, 1, 2, 3, 4)),
    ],
)
def test_splat_iterables(seq, result):
    out = tuple(FortranReader._splat_iterables(*seq))
    assert out == result


@pytest.mark.parametrize(
    "seq, result",
    [
        ((0, slice(1, 4), 4), (slice(0, 5),)),
        ((5, 10, 20), (slice(5, 6), slice(10, 11), slice(20, 21))),
        ((slice(4, 10), slice(10, 20)), (slice(4, 20),)),
    ],
)
def test_combine_sequential(seq, result):
    out = tuple(FortranReader._combine_sequential(*seq))
    assert out == result


@pytest.mark.parametrize(
    "seq, result",
    [
        ((slice(0, 5),), (slice(0, 5),)),
        ((slice(0, 2), slice(4, 6), slice(6, 30)), ([0, 1, 4, 5], slice(6, 30))),
        # The following output is actually undesired : slice(25, 28) should ideally remain as slice
        ((slice(10), slice(25, 28), slice(30, 40)), (slice(0, 10), [25, 26, 27], slice(30, 40))),
    ],
)
def test_slice_combine_array(seq, result):
    out = tuple(FortranReader._slice_combine_array(*seq))
    assert out == result
