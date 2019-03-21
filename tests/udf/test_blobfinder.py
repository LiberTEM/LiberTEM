import numpy as np

import libertem.udf.blobfinder as blobfinder

from utils import MemoryDataSet, _mk_random


def test_refinenemt():
    data = np.array([
        (0, 0, 0, 0, 0, 1),
        (0, 1, 0, 0, 1, 0),
        (0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0),
        (2, 3, 0, 0, 0, 0),
        (0, 2, 0, 0, 0, -10)
    ])

    assert np.allclose(blobfinder.refine_center(center=(1, 1), r=1, corrmap=data), (1, 1))
    assert np.allclose(blobfinder.refine_center(center=(2, 2), r=1, corrmap=data), (1, 1))
    assert np.allclose(blobfinder.refine_center(center=(1, 4), r=1, corrmap=data), (0.5, 4.5))

    y, x = (4, 1)
    ry, rx = blobfinder.refine_center(center=(y, x), r=1, corrmap=data)
    assert (ry > y) and (ry < (y + 1))
    assert (rx < x) and (rx > (x - 1))

    y, x = (4, 4)
    ry, rx = blobfinder.refine_center(center=(y, x), r=1, corrmap=data)
    assert (ry < y) and (ry > (y - 1))
    assert (rx < x) and (rx > (x - 1))


def test_smoke(lt_ctx):
    """
    just check if the analysis rnus without throwing exceptions:
    """
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            partition_shape=(4, 16, 16), sig_dims=2)
    blobfinder.run_blobfinder(ctx=lt_ctx, dataset=dataset, parameters={
        'num_disks': 1,
        'radius': 4,
        'padding': 0,
        'mask_type': 'radial_gradient',
    })


def test_crop_disks_from_frame():
    mask = blobfinder.RadialGradient({
        'radius': 2,
        'padding': 0,
    })
    peaks = [
        [0, 0],
        [2, 2],
        [5, 5],
    ]
    frame = _mk_random(size=(6, 6), dtype="float32")
    crop_disks = list(blobfinder.crop_disks_from_frame(
        peaks,
        frame,
        mask
    ))

    #
    # how is the region around the peak cropped? like this (x denotes the peak position),
    # this is an example for radius 2, padding 0 -> crop_size = 4
    #
    # ---------
    # | | | | |
    # |-|-|-|-|
    # | | | | |
    # |-|-|-|-|
    # | | |x| |
    # |-|-|-|-|
    # | | | | |
    # ---------

    # first peak: top-leftmost; only the bottom right part of the crop_buf should be filled:
    assert crop_disks[0][1] == (slice(2, 4), slice(2, 4))

    # second peak: the whole crop area fits into the frame -> use full crop_buf
    assert crop_disks[1][1] == (slice(0, 4), slice(0, 4))

    # third peak: bottom-rightmost; almost-symmetric to first case
    assert crop_disks[2][1] == (slice(0, 3), slice(0, 3))
