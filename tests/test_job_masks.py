import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from libertem.common.slice import Slice
from libertem.job.masks import MaskContainer, ResultTile, ApplyMasksJob
from libertem.io.dataset.base import DataTile
from libertem.executor.inline import InlineJobExecutor
from libertem.masks import gradient_x
from utils import MemoryDataSet, _naive_mask_apply


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

    dest_slice = res_tile._get_dest_slice_3d(result.shape)
    assert dest_slice[0] == Ellipsis
    # actually let's not test the implementation details here:
    # assert dest_slice[1] == slice(2, 3, None)
    # assert dest_slice[2] == slice(2, 6, None)
    # assert len(dest_slice) == 3

    # let's see if we can select the right slice:
    assert result[..., 2:3, 2:6].shape == (3, 1, 4)

    # the region selected above should be 2:
    assert np.all(result[..., 2:3, 2:6] == 2)

    # everything else should be 0:
    assert np.all(result[..., 2:3, :2] == 0)
    assert np.all(result[..., 2:3, 6:] == 0)
    assert np.all(result[..., :2, :] == 0)
    assert np.all(result[..., 3:, :] == 0)


def test_single_frame_tiles():
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    mask_factories = [
        lambda: mask,
    ]
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16), partition_shape=(16, 16, 16, 16))
    job = ApplyMasksJob(dataset=dataset, mask_factories=mask_factories)

    executor = InlineJobExecutor()

    result = np.zeros((1, 16, 16))
    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(result)

    assert np.allclose(
        result,
        expected
    )


def test_subframe_tiles():
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    mask_factories = [
        lambda: mask,
    ]
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 4, 4), partition_shape=(16, 16, 16, 16))
    job = ApplyMasksJob(dataset=dataset, mask_factories=mask_factories)

    part = next(dataset.get_partitions())

    executor = InlineJobExecutor()

    result = np.zeros((1, 16, 16))
    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(result)

    print(part.shape)
    print(expected)
    print(result)
    assert np.allclose(
        result,
        expected
    )


def test_4d_tilesize():
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    mask_factories = [
        lambda: mask,
    ]
    dataset = MemoryDataSet(data=data, tileshape=(4, 4, 4, 4), partition_shape=(16, 16, 16, 16))
    job = ApplyMasksJob(dataset=dataset, mask_factories=mask_factories)

    executor = InlineJobExecutor()

    result = np.zeros((1, 16, 16))
    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(result)

    assert np.allclose(
        result,
        expected
    )


def test_multirow_tileshape():
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16))
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    mask_factories = [
        lambda: mask,
    ]
    dataset = MemoryDataSet(data=data, tileshape=(4, 16, 16, 16), partition_shape=(16, 16, 16, 16))
    job = ApplyMasksJob(dataset=dataset, mask_factories=mask_factories)

    executor = InlineJobExecutor()

    result = np.zeros((1, 16, 16))
    for tiles in executor.run_job(job):
        for tile in tiles:
            tile.copy_to_result(result)

    assert np.allclose(
        result,
        expected
    )
