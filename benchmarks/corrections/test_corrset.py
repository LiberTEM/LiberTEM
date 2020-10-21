import functools

import numpy as np
import sparse
import pytest

from libertem.corrections import CorrectionSet, detector


@pytest.mark.parametrize(
    "base_shape", ((1, 1), (2, 2))
)
@pytest.mark.parametrize(
    "excluded_coords", (
        # These magic numbers are "worst case" to produce collisions
        # 2*3*4*5*6*7
        np.array([
            (720, 210, 306),
            (120, 210, 210)
        ]),
        # Diagonal that prevents any tiling
        np.array([
            range(1024),
            range(1024),
        ]),
        # Column that prevents tiling in one dimension
        np.array([
            range(1024),
            np.full(1024, 3),
        ])
    )
)
def test_tileshape_adjustment_bench(benchmark, base_shape, excluded_coords):
    sig_shape = (1024, 1024)
    tile_shape = base_shape
    excluded_pixels = sparse.COO(coords=excluded_coords, shape=sig_shape, data=True)
    corr = CorrectionSet(excluded_pixels=excluded_pixels)
    adjusted = benchmark(
        corr.adjust_tileshape,
        tile_shape=tile_shape, sig_shape=sig_shape, base_shape=base_shape
    )
    print("Base shape", base_shape)
    print("Excluded coords", excluded_coords)
    print("Adjusted", adjusted)


def _make_data(nav_dims, sig_dims):
    data = np.linspace(
        start=5, stop=30, num=np.prod(nav_dims) * np.prod(sig_dims), dtype=np.float32
    )
    return data.reshape(nav_dims + sig_dims)


def _generate_exclude_pixels(sig_dims, num_excluded):
    '''
    Generate a list of excluded pixels that
    can be reconstructed faithfully from their neighbors
    in a linear gradient dataset
    '''
    if num_excluded == 0:
        return None
    # Map of pixels that can be reconstructed faithfully from neighbors in a linear gradient
    free_map = np.ones(sig_dims, dtype=np.bool)

    # Exclude all border pixels
    for dim in range(len(sig_dims)):
        selector = tuple(slice(None) if i != dim else (0, -1) for i in range(len(sig_dims)))
        free_map[selector] = False

    exclude = []

    while len(exclude) < num_excluded:
        exclude_item = tuple([np.random.randint(low=1, high=s-1) for s in sig_dims])
        print("Exclude item: ", exclude_item)
        if free_map[exclude_item]:
            exclude.append(exclude_item)
            knock_out = tuple(slice(e - 1, e + 2) for e in exclude_item)
            # Remove the neighbors of a bad pixel
            # since that can't be reconstructed faithfully from a linear gradient
            free_map[knock_out] = False

    print("Remaining free pixel map: ", free_map)

    # Transform from list of tuples with length of number of dimensions
    # to array of indices per dimension
    return np.array(exclude).T


def test_detector_parch_large(benchmark):
    nav_dims = (8, 8)
    sig_dims = (1336, 2004)

    data = _make_data(nav_dims, sig_dims)

    exclude = _generate_exclude_pixels(sig_dims=sig_dims, num_excluded=999)

    assert exclude.shape[1] == 999

    damaged_data = data.copy()
    damaged_data[(Ellipsis, *exclude)] = 1e24

    print("Nav dims: ", nav_dims)
    print("Sig dims:", sig_dims)
    print("Exclude: ", exclude)

    benchmark(
        detector.correct,
        buffer=damaged_data,
        excluded_pixels=exclude,
        sig_shape=sig_dims,
        inplace=False
    )


def test_detector_parch_large_numba_compilation(benchmark):
    nav_dims = (8, 8)
    sig_dims = (1336, 2004)

    data = _make_data(nav_dims, sig_dims)

    exclude = _generate_exclude_pixels(sig_dims=sig_dims, num_excluded=999)

    assert exclude.shape[1] == 999

    damaged_data = data.copy()
    damaged_data[(Ellipsis, *exclude)] = 1e24

    print("Nav dims: ", nav_dims)
    print("Sig dims:", sig_dims)
    print("Exclude: ", exclude)

    benchmark.pedantic(
        functools.partial(
            detector.correct,
            buffer=damaged_data,
            excluded_pixels=exclude,
            sig_shape=sig_dims,
            inplace=False
        ),
        warmup_rounds=0,
        rounds=2,
        iterations=1,
    )
