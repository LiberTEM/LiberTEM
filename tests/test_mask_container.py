import numpy as np

from libertem.job.masks import MaskContainer
from libertem.common import Slice, Shape


def test_mask_caching_1():
    input_masks = [
        lambda: np.ones((128, 128)),
        lambda: np.zeros((128, 128)),
    ]
    mask_container = MaskContainer(mask_factories=input_masks, dtype="float32")

    shape = Shape((16, 16, 128, 128), sig_dims=2)
    slice_ = Slice(origin=(0, 0, 0, 0), shape=shape)
    mask_container[slice_]

    cache_info = mask_container._get_masks_for_slice.cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 1

    mask_container[slice_]

    cache_info = mask_container._get_masks_for_slice.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1

    slice_ = Slice(origin=(0, 1, 0, 0), shape=shape)

    mask_container[slice_]

    cache_info = mask_container._get_masks_for_slice.cache_info()
    assert cache_info.hits == 2
    assert cache_info.misses == 1


def test_mask_caching_2():
    input_masks = [
        lambda: np.ones((128, 128)),
        lambda: np.zeros((128, 128)),
    ]
    mask_container = MaskContainer(mask_factories=input_masks, dtype="float32")

    shape1 = Shape((16, 16, 128, 128), sig_dims=2)
    shape2 = Shape((8, 16, 128, 128), sig_dims=2)
    slice_ = Slice(origin=(0, 0, 0, 0), shape=shape1)
    mask_container[slice_]

    cache_info = mask_container._get_masks_for_slice.cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 1

    mask_container[slice_]

    cache_info = mask_container._get_masks_for_slice.cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1

    slice_ = Slice(origin=(0, 1, 0, 0), shape=shape2)

    mask_container[slice_]

    cache_info = mask_container._get_masks_for_slice.cache_info()
    assert cache_info.hits == 2
    assert cache_info.misses == 1
