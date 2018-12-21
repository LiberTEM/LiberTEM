import pytest
import numpy as np
from scipy.ndimage import measurements
from libertem import masks
from utils import MemoryDataSet, _mk_random


@pytest.fixture
def ds_w_zero_frame():
    data = _mk_random(size=(16, 16, 16, 16))
    data[0, 0] = np.zeros((16, 16))
    dataset = MemoryDataSet(
        data=data.astype("<u2"),
        tileshape=(1, 1, 16, 16),
        partition_shape=(8, 16, 16, 16)
    )
    return dataset


@pytest.fixture
def ds_random():
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data.astype("<u2"),
        tileshape=(1, 1, 16, 16),
        partition_shape=(16, 16, 16, 16)
    )
    return dataset


def test_com_with_zero_frames(ds_w_zero_frame, lt_ctx):
    analysis = lt_ctx.create_com_analysis(
        dataset=ds_w_zero_frame, cx=0, cy=0, mask_radius=0
    )
    results = lt_ctx.run(analysis)

    # no inf/nan in center_x and center_y
    assert not np.any(np.isinf(results[3].raw_data))
    assert not np.any(np.isinf(results[4].raw_data))
    assert not np.any(np.isnan(results[3].raw_data))
    assert not np.any(np.isnan(results[4].raw_data))

    # no inf/nan in divergence/magnitude
    assert not np.any(np.isinf(results[1].raw_data))
    assert not np.any(np.isinf(results[2].raw_data))
    assert not np.any(np.isnan(results[1].raw_data))
    assert not np.any(np.isnan(results[2].raw_data))


def test_com_comparison_scipy_1_nomask(ds_random, lt_ctx):
    analysis = lt_ctx.create_com_analysis(
        dataset=ds_random, cx=0, cy=0, mask_radius=None
    )
    results = lt_ctx.run(analysis)
    raw_data_by_frame = ds_random.data.reshape((16 * 16, 16, 16))
    field_y, field_x = results.field.raw_data
    field_y = field_y.reshape((16 * 16))
    field_x = field_x.reshape((16 * 16))
    for idx in range(16 * 16):
        scx, scy = measurements.center_of_mass(raw_data_by_frame[idx])
        assert np.allclose(scx, field_x[idx])
        assert np.allclose(scy, field_y[idx])


def test_com_comparison_scipy_2_masked(ds_random, lt_ctx):
    analysis = lt_ctx.create_com_analysis(
        dataset=ds_random, cx=0, cy=0, mask_radius=8
    )
    results = lt_ctx.run(analysis)
    raw_data_by_frame = ds_random.data.reshape((16 * 16, 16, 16))
    field_y, field_x = results.field.raw_data
    field_y = field_y.reshape((16 * 16))
    field_x = field_x.reshape((16 * 16))
    disk_mask = masks.circular(
        centerX=0, centerY=0,
        imageSizeX=16,
        imageSizeY=16,
        radius=8,
    )
    for idx in range(16 * 16):
        masked_frame = raw_data_by_frame[idx] * disk_mask
        scx, scy = measurements.center_of_mass(masked_frame)
        assert np.allclose(scx, field_x[idx])
        assert np.allclose(scy, field_y[idx])


def test_com_fails_with_non_4d_data_1(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16))
    dataset = MemoryDataSet(
        data=data.astype("<u2"),
        tileshape=(1, 16, 16),
        partition_shape=(8, 16, 16)
    )
    with pytest.raises(Exception):
        lt_ctx.create_com_analysis(
            dataset=dataset, cx=0, cy=0, mask_radius=8
        )


def test_com_fails_with_non_4d_data_2(lt_ctx):
    data = _mk_random(size=(16, 16, 16 * 16))
    dataset = MemoryDataSet(
        data=data.astype("<u2"),
        tileshape=(1, 16, 16),
        partition_shape=(16, 16, 16),
        sig_dims=1,
    )
    with pytest.raises(Exception):
        lt_ctx.create_com_analysis(
            dataset=dataset, cx=0, cy=0, mask_radius=8
        )
