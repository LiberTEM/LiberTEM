import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from libertem.common.slice import Slice
from libertem.job.masks import MaskContainer, ResultTile
from libertem.io.dataset.base import DataTile
from libertem.masks import gradient_x


@pytest.fixture
def masks():
    input_masks = [
        lambda: np.ones((128, 128)),
        lambda: np.zeros((128, 128)),
        lambda: np.ones((128, 128)),
        lambda: gradient_x(128, 128, dtype=np.float32),
    ]
    return MaskContainer(mask_factories=input_masks, dtype=np.float32)


def test_merge_masks(masks):
    assert masks.shape == (128 * 128, 4)


def test_for_datatile_1(masks):
    tile = DataTile(
        tile_slice=Slice(origin=(0, 0, 0, 0), shape=(1, 1, 1, 1)),
        data=np.ones((1, 1, 1, 1))
    )
    slice_ = masks.get_masks_for_slice(tile.tile_slice)
    assert slice_.shape == (1, 4)


def test_for_datatile_2(masks):
    tile = DataTile(
        tile_slice=Slice(origin=(0, 0, 0, 0), shape=(2, 2, 10, 10)),
        data=np.ones((2, 2, 10, 10))
    )
    slice_ = masks.get_masks_for_slice(tile.tile_slice)
    assert slice_.shape == (100, 4)


def test_for_datatile_with_scan_origin(masks):
    tile = DataTile(
        tile_slice=Slice(origin=(10, 10, 0, 0), shape=(2, 2, 10, 10)),
        data=np.ones((2, 2, 10, 10))
    )
    slice_ = masks.get_masks_for_slice(tile.tile_slice)
    assert slice_.shape == (100, 4)


def test_for_datatile_with_frame_origin(masks):
    tile = DataTile(
        tile_slice=Slice(origin=(10, 10, 10, 10), shape=(2, 2, 1, 5)),
        data=np.ones((2, 2, 1, 5))
    )
    slice_ = masks.get_masks_for_slice(tile.tile_slice)
    print(slice_)
    assert_array_almost_equal(
        slice_,
        np.array([
            1, 0, 1, 10,
            1, 0, 1, 11,
            1, 0, 1, 12,
            1, 0, 1, 13,
            1, 0, 1, 14,
        ]).reshape((5, 4))
    )


def test_copy_to_result():
    # result tile: for three masks, insert all ones into the given position:
    res_tile = ResultTile(
        data=np.ones((
            4,  # xdim*ydim, flattened
            3,  # num masks
        )),
        tile_slice=Slice(origin=(2, 2, 0, 0), shape=(1, 4, 10, 10)),
    )
    result = np.zeros(
        (3,     # num masks
         10,    # ydim
         10)    # xdim
    )
    res_tile.copy_to_result(result)
    res_tile.copy_to_result(result)
    print(result)

    dest_slice = res_tile._get_dest_slice()
    assert dest_slice[0] == Ellipsis
    assert dest_slice[1] == slice(2, 3, None)
    assert dest_slice[2] == slice(2, 6, None)

    assert len(dest_slice) == 3

    # let's see if we can select the right slice:
    assert result[..., 2:3, 2:6].shape == (3, 1, 4)

    # the region selected above should be 2:
    assert np.all(result[..., 2:3, 2:6] == 2)

    # everything else should be 0:
    assert np.all(result[..., 2:3, :2] == 0)
    assert np.all(result[..., 2:3, 6:] == 0)
    assert np.all(result[..., :2, :] == 0)
    assert np.all(result[..., 3:, :] == 0)
