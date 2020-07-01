import numpy as np
import sparse

import pytest

from libertem.corrections import CorrectionSet


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
    sig_shape = (1014, 1024)
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
