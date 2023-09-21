import numpy as np
import scipy.sparse as sp
import sparse
import pytest
from sparseconverter import CUPY

from libertem.common.container import MaskContainer
from libertem.io.dataset.base import DataTile
from libertem.common import Slice, Shape
from libertem.masks import gradient_x


@pytest.fixture
def masks():
    input_masks = [
        lambda: np.ones((128, 128)),
        lambda: sparse.zeros((128, 128)),
        lambda: np.ones((128, 128)),
        lambda: sp.csr_matrix(((1,), ((64,), (64,))), shape=(128, 128), dtype=np.float32),
        lambda: gradient_x(128, 128, dtype=np.float32),
    ]
    return MaskContainer(mask_factories=input_masks, dtype=np.float32)


def test_mask_caching_1():
    input_masks = [
        lambda: np.ones((128, 128)),
        lambda: np.zeros((128, 128)),
    ]
    mask_container = MaskContainer(mask_factories=input_masks, dtype="float32")

    shape = Shape((16 * 16, 128, 128), sig_dims=2)
    slice_ = Slice(origin=(0, 0, 0), shape=shape)
    mask_container.get(slice_)

    key = (mask_container.dtype, False, True, 'numpy')

    cache_info = mask_container._get_masks_for_slice[key].cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 1

    mask_container.get(slice_)

    cache_info = mask_container._get_masks_for_slice[key].cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1

    slice_ = Slice(origin=(1, 0, 0), shape=shape)

    mask_container.get(slice_)

    cache_info = mask_container._get_masks_for_slice[key].cache_info()
    assert cache_info.hits == 2
    assert cache_info.misses == 1


def test_mask_caching_2():
    input_masks = [
        lambda: np.ones((128, 128)),
        lambda: np.zeros((128, 128)),
    ]
    mask_container = MaskContainer(mask_factories=input_masks, dtype="float32")

    shape1 = Shape((16 * 16, 128, 128), sig_dims=2)
    shape2 = Shape((8 * 16, 128, 128), sig_dims=2)
    slice_ = Slice(origin=(0, 0, 0), shape=shape1)
    mask_container.get(slice_)

    key = (mask_container.dtype, False, True, 'numpy')

    cache_info = mask_container._get_masks_for_slice[key].cache_info()
    assert cache_info.hits == 0
    assert cache_info.misses == 1

    mask_container.get(slice_)

    cache_info = mask_container._get_masks_for_slice[key].cache_info()
    assert cache_info.hits == 1
    assert cache_info.misses == 1

    slice_ = Slice(origin=(1, 0, 0), shape=shape2)

    mask_container.get(slice_)

    cache_info = mask_container._get_masks_for_slice[key].cache_info()
    assert cache_info.hits == 2
    assert cache_info.misses == 1


def test_for_datatile_1(masks):
    tile = DataTile(
        np.ones((1, 1, 1)),
        tile_slice=Slice(origin=(0, 0, 0), shape=Shape((1, 1, 1), sig_dims=2)),
        scheme_idx=0,
    )
    slice_ = masks.get_masks_for_slice(tile.tile_slice)
    assert slice_.shape == (1, 5)


def test_for_datatile_2(masks):
    tile = DataTile(
        np.ones((2 * 2, 10, 10)),
        tile_slice=Slice(origin=(0, 0, 0), shape=Shape((2 * 2, 10, 10), sig_dims=2)),
        scheme_idx=0,
    )
    slice_ = masks.get_masks_for_slice(tile.tile_slice)
    assert slice_.shape == (100, 5)


def test_for_datatile_with_scan_origin(masks):
    tile = DataTile(
        np.ones((2 * 2, 10, 10)),
        tile_slice=Slice(origin=(110, 0, 0), shape=Shape((2 * 2, 10, 10), sig_dims=2)),
        scheme_idx=0,
    )
    slice_ = masks.get_masks_for_slice(tile.tile_slice)
    assert slice_.shape == (100, 5)


def test_for_datatile_with_frame_origin(masks):
    tile = DataTile(
        np.ones((2 * 2, 1, 5)),
        tile_slice=Slice(origin=(110, 10, 10), shape=Shape((2 * 2, 1, 5), sig_dims=2)),
        scheme_idx=0,
    )
    slice_ = masks.get_masks_for_slice(tile.tile_slice)
    print(slice_)
    np.allclose(
        slice_,
        np.array([
            1, 0, 1, 0, 10,
            1, 0, 1, 0, 11,
            1, 0, 1, 0, 12,
            1, 0, 1, 0, 13,
            1, 0, 1, 0, 14,
        ]).reshape((5, 5))
    )


def test_merge_masks(masks):
    assert masks.computed_masks.shape == (5, 128, 128)


def test_sparse_pydata_cupy_default_unsupported():
    input_masks = [
        lambda: np.ones((128, 128)),
    ]
    factory = MaskContainer(
        mask_factories=input_masks,
        use_sparse=None,
        default_sparse='sparse.pydata',
        backend=CUPY,
    )
    assert factory.use_sparse is False


def test_raises_unknown_use_sparse():
    input_masks = [
        lambda: np.ones((128, 128)),
    ]
    with pytest.raises(ValueError):
        MaskContainer(
            mask_factories=input_masks,
            use_sparse='unknown',
        )
