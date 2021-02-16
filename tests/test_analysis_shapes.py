import pytest
import numpy as np

from libertem.io.dataset.memory import MemoryDataSet

from utils import _naive_mask_apply, _mk_random


@pytest.fixture
def ds_random():
    data = _mk_random(size=(16, 16, 16, 16))
    dataset = MemoryDataSet(
        data=data.astype("<u2"),
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2,
    )
    return dataset


def test_disk_1(lt_ctx, ds_random):
    analysis = lt_ctx.create_disk_analysis(dataset=ds_random, cx=8, cy=8, r=5)
    results = lt_ctx.run(analysis)
    mask = analysis.get_mask_factories()[0]()
    expected = _naive_mask_apply([mask], ds_random.data)
    # TODO: test the actual mask shape, too
    # at least mask application should match:
    assert np.allclose(
        results.intensity.raw_data,
        expected,
    )
    assert np.allclose(
        results.intensity_log.raw_data,
        expected,
    )


def test_disk_defaults(lt_ctx, ds_random):
    analysis = lt_ctx.create_disk_analysis(dataset=ds_random)
    results = lt_ctx.run(analysis)
    mask = analysis.get_mask_factories()[0]()
    expected = _naive_mask_apply([mask], ds_random.data)
    # TODO: test the actual mask shape, too
    # at least mask application should match:
    assert np.allclose(
        results.intensity.raw_data,
        expected,
    )
    assert np.allclose(
        results.intensity_log.raw_data,
        expected,
    )


def test_ring_1(lt_ctx, ds_random):
    analysis = lt_ctx.create_ring_analysis(dataset=ds_random, cx=8, cy=8, ri=5, ro=8)
    results = lt_ctx.run(analysis)
    mask = analysis.get_mask_factories()[0]()
    expected = _naive_mask_apply([mask], ds_random.data)
    assert results.intensity.raw_data.shape == (16, 16)
    assert np.allclose(
        results.intensity.raw_data,
        expected,
    )
    assert np.allclose(
        results.intensity_log.raw_data,
        expected,
    )


def test_ring_3d_ds(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16))
    dataset = MemoryDataSet(
        data=data.astype("<u2"),
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2,
    )
    analysis = lt_ctx.create_ring_analysis(dataset=dataset, cx=8, cy=8, ri=5, ro=8)
    results = lt_ctx.run(analysis)
    mask = analysis.get_mask_factories()[0]()
    expected = _naive_mask_apply([mask], dataset.data.reshape((16, 16, 16, 16)))
    assert results.intensity.raw_data.shape == (16 * 16,)
    assert np.allclose(
        results.intensity.raw_data.reshape((16, 16)),
        expected,
    )
    assert np.allclose(
        results.intensity_log.raw_data.reshape((16, 16)),
        expected,
    )


def test_ring_defaults(lt_ctx, ds_random):
    analysis = lt_ctx.create_ring_analysis(dataset=ds_random)
    results = lt_ctx.run(analysis)
    mask = analysis.get_mask_factories()[0]()
    expected = _naive_mask_apply([mask], ds_random.data)
    assert np.allclose(
        results.intensity.raw_data,
        expected,
    )
    assert np.allclose(
        results.intensity_log.raw_data,
        expected,
    )


def test_point_1(lt_ctx, ds_random):
    analysis = lt_ctx.create_point_analysis(dataset=ds_random, x=8, y=8)
    results = lt_ctx.run(analysis)
    mask = np.zeros((16, 16))
    mask[8, 8] = 1
    expected = _naive_mask_apply([mask], ds_random.data)
    assert np.allclose(
        results.intensity.raw_data,
        expected,
    )
    assert np.allclose(
        results.intensity_log.raw_data,
        expected,
    )


def test_point_3d_ds(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16))
    dataset = MemoryDataSet(
        data=data.astype("<u2"),
        tileshape=(1, 16, 16),
        num_partitions=2,
        sig_dims=2,
    )
    analysis = lt_ctx.create_point_analysis(dataset=dataset, x=8, y=8)
    results = lt_ctx.run(analysis)
    mask = np.zeros((16, 16))
    mask[8, 8] = 1
    expected = _naive_mask_apply([mask], dataset.data.reshape((16, 16, 16, 16)))
    assert results.intensity.raw_data.shape == (16 * 16,)
    assert np.allclose(
        results.intensity.raw_data.reshape((16, 16)),
        expected,
    )
    assert np.allclose(
        results.intensity_log.raw_data.reshape((16, 16)),
        expected,
    )


def test_point_defaults(lt_ctx, ds_random):
    analysis = lt_ctx.create_point_analysis(dataset=ds_random)
    results = lt_ctx.run(analysis)
    mask = np.zeros((16, 16))
    mask[8, 8] = 1
    expected = _naive_mask_apply([mask], ds_random.data)
    assert results.intensity.raw_data.shape == (16, 16)
    assert np.allclose(
        results.intensity.raw_data,
        expected,
    )
    assert np.allclose(
        results.intensity_log.raw_data,
        expected,
    )


def test_disk_complex(lt_ctx, ds_complex):
    analysis = lt_ctx.create_disk_analysis(dataset=ds_complex)
    results = lt_ctx.run(analysis)
    mask = analysis.get_mask_factories()[0]()
    expected = _naive_mask_apply([mask], ds_complex.data)
    assert np.allclose(
        results.intensity_complex.raw_data,
        expected,
    )
    assert np.allclose(
        results.intensity_log.raw_data,
        np.abs(results.intensity_complex),
    )
    assert np.allclose(
        results.intensity.raw_data,
        np.abs(results.intensity_complex),
    )


def test_ring_complex(lt_ctx, ds_complex):
    analysis = lt_ctx.create_ring_analysis(dataset=ds_complex)
    results = lt_ctx.run(analysis)
    mask = analysis.get_mask_factories()[0]()
    expected = _naive_mask_apply([mask], ds_complex.data)
    assert np.allclose(
        results.intensity_complex.raw_data,
        expected,
    )
    assert np.allclose(
        results.intensity_log.raw_data,
        np.abs(results.intensity_complex),
    )
    assert np.allclose(
        results.intensity.raw_data,
        np.abs(results.intensity_complex),
    )


def test_point_complex(lt_ctx, ds_complex):
    analysis = lt_ctx.create_point_analysis(dataset=ds_complex)
    results = lt_ctx.run(analysis)
    mask = analysis.get_mask_factories()[0]()
    expected = _naive_mask_apply([mask], ds_complex.data)
    assert np.allclose(
        results.intensity_complex.raw_data,
        expected,
    )
    assert np.allclose(
        results.intensity_log.raw_data,
        np.abs(results.intensity_complex),
    )
    assert np.allclose(
        results.intensity.raw_data,
        np.abs(results.intensity_complex),
    )
