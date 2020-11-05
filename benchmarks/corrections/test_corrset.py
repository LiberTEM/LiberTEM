import numpy as np
import sparse
import pytest

from libertem.corrections import CorrectionSet, detector
from libertem.utils.generate import gradient_data, exclude_pixels


@pytest.mark.benchmark(
    group="adjust tileshape",
)
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


@pytest.mark.benchmark(
    group="patch many",
)
@pytest.mark.parametrize(
    "num_excluded", (0, 1, 10, 100, 1000, 10000)
)
def test_detector_patch_large(num_excluded, benchmark):
    nav_dims = (8, 8)
    sig_dims = (1336, 2004)

    data = gradient_data(nav_dims, sig_dims)

    exclude = exclude_pixels(sig_dims=sig_dims, num_excluded=num_excluded)

    damaged_data = data.copy()

    if exclude is not None:
        assert exclude.shape[1] == num_excluded
        damaged_data[(Ellipsis, *exclude)] = 1e24

    print("Nav dims: ", nav_dims)
    print("Sig dims:", sig_dims)
    print("Exclude: ", exclude)

    benchmark.pedantic(
        detector.correct,
        kwargs=dict(
            buffer=damaged_data,
            excluded_pixels=exclude,
            sig_shape=sig_dims,
            inplace=False
        ),
        warmup_rounds=0,
        rounds=5,
        iterations=1,
    )


@pytest.mark.benchmark(
    group="correct large",
)
def test_detector_correction_large(benchmark):
    nav_dims = (8, 8)
    sig_dims = (1336, 2004)

    data = gradient_data(nav_dims, sig_dims)
    gain_map = (np.random.random(sig_dims) + 1).astype(np.float64)
    dark_image = np.random.random(sig_dims).astype(np.float64)

    damaged_data = data.copy()
    damaged_data /= gain_map
    damaged_data += dark_image

    print("Nav dims: ", nav_dims)
    print("Sig dims:", sig_dims)

    benchmark.pedantic(
        detector.correct,
        kwargs=dict(
            buffer=damaged_data,
            dark_image=dark_image,
            gain_map=gain_map,
            sig_shape=sig_dims,
            inplace=False
        ),
        warmup_rounds=0,
        rounds=5,
        iterations=1,
    )
