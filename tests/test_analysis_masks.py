import pytest
import numpy as np
import scipy.sparse as sp
from numpy.testing import assert_array_almost_equal
from libertem.common.slice import Slice
from libertem.job.masks import MaskContainer
from libertem.io.dataset.base import DataTile
from libertem.masks import gradient_x, to_dense, to_sparse
from utils import MemoryDataSet, _naive_mask_apply

def _run_mask_test_program(lt_ctx, dataset, mask, expected):
    analysis_default = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask]
    )
    analysis_sparse = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: to_sparse(mask)], use_sparse=True
    )
    analysis_dense = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: to_dense(mask)], use_sparse=False
    )
    results_default = lt_ctx.run(analysis_default)
    results_sparse = lt_ctx.run(analysis_sparse)
    results_dense = lt_ctx.run(analysis_dense)

    assert np.allclose(
        results_default.mask_0.raw_data,
        expected
    )
    assert np.allclose(
        results_sparse.mask_0.raw_data,
        expected
    )
    assert np.allclose(
        results_dense.mask_0.raw_data,
        expected
    )

@pytest.fixture
def masks():
    input_masks = [
        lambda: np.ones((128, 128)),
        lambda: np.zeros((128, 128)),
        lambda: np.ones((128, 128)),
        lambda: sp.csr_matrix(((1,), ((64,), (64,))), shape=(128, 128), dtype=np.float32),
        lambda: gradient_x(128, 128, dtype=np.float32),
    ]
    return MaskContainer(mask_factories=input_masks, dtype=np.float32)


def test_merge_masks(masks):
    assert masks.shape == (128 * 128, 5)


def test_for_datatile_1(masks):
    tile = DataTile(
        tile_slice=Slice(origin=(0, 0, 0, 0), shape=(1, 1, 1, 1)),
        data=np.ones((1, 1, 1, 1))
    )
    slice_ = masks.get_masks_for_slice(tile.tile_slice)
    assert slice_.shape == (1, 5)


def test_for_datatile_2(masks):
    tile = DataTile(
        tile_slice=Slice(origin=(0, 0, 0, 0), shape=(2, 2, 10, 10)),
        data=np.ones((2, 2, 10, 10))
    )
    slice_ = masks.get_masks_for_slice(tile.tile_slice)
    assert slice_.shape == (100, 5)


def test_for_datatile_with_scan_origin(masks):
    tile = DataTile(
        tile_slice=Slice(origin=(10, 10, 0, 0), shape=(2, 2, 10, 10)),
        data=np.ones((2, 2, 10, 10))
    )
    slice_ = masks.get_masks_for_slice(tile.tile_slice)
    assert slice_.shape == (100, 5)


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
            1, 0, 1, 0, 10,
            1, 0, 1, 0, 11,
            1, 0, 1, 0, 12,
            1, 0, 1, 0, 13,
            1, 0, 1, 0, 14,
        ]).reshape((5, 5))
    )


def test_weird_partition_shapes_1(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16), partition_shape=(16, 16, 2, 2))

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_single_frame_tiles(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16), partition_shape=(16, 16, 16, 16))

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_subframe_tiles(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 4, 4), partition_shape=(16, 16, 16, 16))

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_4d_tilesize(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(4, 4, 4, 4), partition_shape=(16, 16, 16, 16))
    
    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_multirow_tileshape(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(4, 16, 16, 16), partition_shape=(16, 16, 16, 16))
    
    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_mask_uint(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask = np.random.choice(a=[0, 1], size=(16, 16)).astype("uint16")
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(4, 4, 4, 4), partition_shape=(16, 16, 16, 16))
    
    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_endian(lt_ctx):
    data = np.random.choice(a=0xFFFF, size=(16, 16, 16, 16)).astype(">u2")
    mask = np.random.choice(a=[0, 1], size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(4, 4, 4, 4), partition_shape=(16, 16, 16, 16))
    
    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_multi_masks(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask0 = np.random.choice(a=[0, 1], size=(16, 16))
    mask1 = sp.csr_matrix(np.random.choice(a=[0, 1], size=(16, 16)))
    expected = _naive_mask_apply([mask0, mask1], data)

    dataset = MemoryDataSet(data=data, tileshape=(4, 4, 4, 4), partition_shape=(16, 16, 16, 16))
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1]
    )
    results = lt_ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1.raw_data,
        expected[1],
    )


def test_mask_job(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask0 = np.random.choice(a=[0, 1], size=(16, 16))
    mask1 = sp.csr_matrix(np.random.choice(a=[0, 1], size=(16, 16)))
    expected = _naive_mask_apply([mask0, mask1], data)

    dataset = MemoryDataSet(data=data, tileshape=(4, 4, 4, 4), partition_shape=(16, 16, 16, 16))
    job = lt_ctx.create_mask_job(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1]
    )
    results = lt_ctx.run(job)

    assert np.allclose(
        results,
        expected,
    )


def test_all_sparse_analysis(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask0 = sp.csr_matrix(np.random.choice(a=[0, 1], size=(16, 16)))
    mask1 = sp.csr_matrix(np.random.choice(a=[0, 1], size=(16, 16)))
    expected = _naive_mask_apply([mask0, mask1], data)

    dataset = MemoryDataSet(data=data, tileshape=(4, 4, 4, 4), partition_shape=(16, 16, 16, 16))
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1]
    )
    results = lt_ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1.raw_data,
        expected[1],
    )


def test_uses_sparse_all_default(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask0 = sp.csr_matrix(np.random.choice(a=[0, 1], size=(16, 16)))
    mask1 = sp.csr_matrix(np.random.choice(a=[0, 1], size=(16, 16)))

    dataset = MemoryDataSet(data=data, tileshape=(4, 4, 4, 4), partition_shape=(16, 16, 16, 16))
    job = lt_ctx.create_mask_job(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1]
    )
    
    tiles = job.dataset.get_partitions()
    tile = next(tiles)

    assert sp.issparse(job.masks[tile])


def test_uses_sparse_mixed_default(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask0 = sp.csr_matrix(np.random.choice(a=[0, 1], size=(16, 16)))
    mask1 = np.random.choice(a=[0, 1], size=(16, 16))

    dataset = MemoryDataSet(data=data, tileshape=(4, 4, 4, 4), partition_shape=(16, 16, 16, 16))
    job = lt_ctx.create_mask_job(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1]
    )
    tiles = job.dataset.get_partitions()
    tile = next(tiles)

    assert not sp.issparse(job.masks[tile])


def test_uses_sparse_true(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask0 = np.random.choice(a=[0, 1], size=(16, 16))
    mask1 = np.random.choice(a=[0, 1], size=(16, 16))

    dataset = MemoryDataSet(data=data, tileshape=(4, 4, 4, 4), partition_shape=(16, 16, 16, 16))
    job = lt_ctx.create_mask_job(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1], use_sparse=True
    )

    tiles = job.dataset.get_partitions()
    tile = next(tiles)

    assert sp.issparse(job.masks[tile])


def test_uses_sparse_false(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    mask0 = sp.csr_matrix(np.random.choice(a=[0, 1], size=(16, 16)))
    mask1 = sp.csr_matrix(np.random.choice(a=[0, 1], size=(16, 16)))

    dataset = MemoryDataSet(data=data, tileshape=(4, 4, 4, 4), partition_shape=(16, 16, 16, 16))
    job = lt_ctx.create_mask_job(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1], use_sparse=False
    )
    tiles = job.dataset.get_partitions()
    tile = next(tiles)

    assert not sp.issparse(job.masks[tile])
