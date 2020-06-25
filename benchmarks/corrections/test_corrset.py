import numpy as np
import sparse

from libertem.corrections import CorrectionSet


def test_tileshape_adjustment_bench(benchmark):
    sig_shape = (1014, 1024)
    tile_shape = (1, 1)
    base_shape = (1, 1)
    # These magic numbers are "worst case" to produce collisions
    # 2*3*4*5*6*7
    excluded_coords = np.array([
        (720, 210, 306),
        (120, 210, 210)
    ])
    excluded_pixels = sparse.COO(coords=excluded_coords, shape=sig_shape, data=True)
    corr = CorrectionSet(excluded_pixels=excluded_pixels)
    adjusted = benchmark(
        corr.adjust_tileshape,
        tile_shape=tile_shape, sig_shape=sig_shape, base_shape=base_shape
    )
    print(adjusted)
    assert adjusted != (1014, 1024)
