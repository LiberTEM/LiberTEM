import itertools
import numpy as np

import pytest

from libertem.common.shape import Shape
from libertem.io.dataset.base.tiling_scheme import TilingScheme
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


@pytest.mark.parametrize(
    "ideal_depth, ts_depth, sl",
    [
        (1, 3, slice(5, 100)),
        (16, 1, slice(5, 6)),
        (16, 8, slice(5, 6)),
    ],
)
def test_plan_reads_buffer_length(ideal_depth, ts_depth, sl):
    buffer_length, _ = FortranReader._plan_reads(ideal_depth,
                                                 ts_depth,
                                                 sl)
    if (sl.stop - sl.start) <= ts_depth:
        assert buffer_length == (sl.stop - sl.start)
    else:
        assert buffer_length >= ts_depth
        assert (buffer_length % ts_depth) == 0


def _generate_random_slices():
    start = np.random.randint(0, 5)
    length = np.random.choice([1, 3, 37, 54, 8, 113, 64, 511])
    end = start + length
    contig = np.random.choice([False, True])
    if contig:
        yield slice(start, end)
        return
    _upto = start
    while _upto < end:
        yield_length = np.random.choice([1,
                                         np.random.randint(2, max(3, end - _upto))],
                                        p=(0.4, 0.6))
        if yield_length == 1:
            yield _upto
        else:
            yield slice(_upto, min(_upto + yield_length, end))
        _upto += yield_length
        _upto += np.random.choice([0,
                                   np.random.randint(2, max(3, end - _upto))],
                                  p=(0.1, 0.9))


def _expand_slices(*slices):
    indices = tuple(i if isinstance(i, (np.integer, int))
                    else range(i.start, i.stop)
                    for i in slices)
    return tuple(FortranReader._splat_iterables(*indices))


@pytest.mark.parametrize('repeat', tuple(range(30)))
def test_plan_reads(repeat):
    ideal_depth = np.random.choice([1, 3, 8, 64])
    ts_depth = np.random.choice([1, 3, 8, 16, 128])
    slices = tuple(_generate_random_slices())

    buffer_length, reads = FortranReader._plan_reads(ideal_depth,
                                                     ts_depth,
                                                     *slices)

    # The return value reads has the following structure
    #     list of (tuple(slice,...) for_memmap), slice_for_buffer_alloc, list(unpacks))
    # where:
    #     unpacks == (slice_in_buffer, tuple(int, ...) of flat_frame_idcs)

    # verify sum of all reads covers all idcs from input slices
    assert _expand_slices(*itertools.chain(*tuple(r[0] for r in reads))) == _expand_slices(*slices)
    # verify each read group fits into a single buffer
    assert all(len(_expand_slices(*r[0])) <= buffer_length for r in reads)
    # verify the length of each read is placed into the correct slice size of output buffer
    assert all(len(_expand_slices(*r[0])) == (r[1].stop - r[1].start) for r in reads)
    # verify that each read group is unpacked to the right frame indices
    assert all(_expand_slices(*r[0]) == tuple(itertools.chain(*tuple(rr[1]
                                                                     for rr
                                                                     in r[2])))
               for r in reads)
    # verify all unpacks from buffer are nonzero and ts_depth or less long
    assert all(all(0 < (rr[0].stop - rr[0].start) <= ts_depth for rr in r[2]) for r in reads)
    # verify each unpack unpacks to the correct number of frame indices
    assert all((rr[0].stop - rr[0].start) == len(rr[1]) for r in reads for rr in r[2])


@pytest.mark.parametrize(
    "shape, tileshape, order, raises",
    [
        (Shape((8, 30, 50), 2), Shape((3, 30, 3), 2), 'F', None),
        (Shape((8, 30, 50), 2), Shape((3, 10, 5), 2), 'F', AssertionError),
        (Shape((8, 30, 50), 2), Shape((3, 4, 15), 2), 'C', AssertionError),
        (Shape((8, 4, 30, 50), 3), Shape((3, 1, 50, 50), 3), 'C', None),
        (Shape((1, 2, 3), 2), Shape((1, 2, 3), 2), 'K', ValueError),
    ],
)
def test_verify_tiling(shape, tileshape, order, raises):
    scheme = TilingScheme.make_for_shape(tileshape, shape)
    if raises is not None:
        with pytest.raises(raises):
            FortranReader.verify_tiling(scheme, shape, order)
    else:
        FortranReader.verify_tiling(scheme, shape, order)


@pytest.mark.parametrize(
    "shapes, slices",
    [
        (((10, 20), (10, 30)), (slice(0, 200), slice(200, 500))),
        (((5, 10, 5),), (slice(0, 250),)),
    ],
)
def test_flat_tile_slices(shapes, slices):
    assert FortranReader._flat_tile_slices(shapes) == slices


def test_build_chunk_map1():
    chunk_schemes = [{4}, {5, 6}, {6}]
    unique, combined = FortranReader.build_chunk_map(chunk_schemes)
    assert unique == {0: {4}, 1: {5}, 2: set()}
    assert combined == {(1, 2): {6}}


def test_build_chunk_map2():
    chunk_schemes = [{0}]
    unique, combined = FortranReader.build_chunk_map(chunk_schemes)
    assert unique == {0: {0}}
    assert combined == {}


def test_build_chunk_map3():
    chunk_schemes = [{1}, {4}, {10}, {4}]
    unique, combined = FortranReader.build_chunk_map(chunk_schemes)
    assert unique == {0: {1}, 1: set(), 2: {10}, 3: set()}
    assert combined == {(1, 3): {4}}


def test_build_chunk_map4():
    chunk_schemes = [{0}, {0}, {0}, {0}]
    unique, combined = FortranReader.build_chunk_map(chunk_schemes)
    assert unique == {0: set(), 1: set(), 2: set(), 3: set()}
    assert combined == {(0, 1, 2, 3): {0}}


def test_build_chunk_map_raises():
    chunk_schemes = [{}, {5, 6}, {6}]
    with pytest.raises(ValueError):
        FortranReader.build_chunk_map(chunk_schemes)


def _slice_intersects(sl0: slice, sl1: slice) -> bool:
    ll = max(sl0.start, sl1.start)
    uu = min(sl0.stop, sl1.stop)
    return uu - ll > 0


@pytest.mark.parametrize(
    "ds_size_mb", [30, 500, 3000, 6000, 40000],
)
@pytest.mark.parametrize(
    "num_tiles", [1, 3, 13, 63],
)
def test_choose_chunks(ds_size_mb, num_tiles):
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize
    ds_size = ds_size_mb * 2**20
    sig_shape = tuple(np.random.randint(128, 1036, size=(2,)))
    sig_size = np.prod(sig_shape, dtype=np.int64) * itemsize
    n_frames = ds_size // sig_size
    shape = Shape((n_frames,) + sig_shape, 2)
    tile_width = sig_shape[0] // num_tiles
    tileshape = Shape((min(n_frames, 8), tile_width, sig_shape[-1]), 2)

    tiling_scheme = TilingScheme.make_for_shape(tileshape, shape)
    chunks, chunk_slices, scheme_indices = FortranReader.choose_chunks(tiling_scheme,
                                                                       shape,
                                                                       dtype)

    # Check we meet max num memmap param
    assert len(chunks) <= FortranReader.MAX_NUM_MEMMAP
    # All scheme indices are provided by the chunks
    assert set().union(*scheme_indices) == set(range(len(tiling_scheme)))
    # All scheme chunks cover the full nav_dimension
    assert all(c[-1] == shape[0] for c in chunks)
    # Assert chunk sig components sum to total sig size
    assert sum(c[0] for c in chunks) == shape.sig.size
    # Assert chunk slices are reasonable
    assert chunk_slices[0].start == 0
    assert chunk_slices[-1].stop == shape.sig.size
    assert all(s0.stop == s1.start for s0, s1 in zip(chunk_slices[:-1],
                                                     chunk_slices[1:]))
    # Check each tile slice lies in the indicated chunks
    tile_shapes = tuple(s.shape.to_tuple() for s in tiling_scheme)
    tile_slices = FortranReader._flat_tile_slices(tile_shapes)
    for scheme_idx, sl in enumerate(tile_slices):
        for chunk_slice, chunk_scheme_idcs in zip(chunk_slices, scheme_indices):
            if scheme_idx in chunk_scheme_idcs:
                assert _slice_intersects(chunk_slice, sl)
