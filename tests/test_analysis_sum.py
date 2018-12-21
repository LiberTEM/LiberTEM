import numpy as np

from utils import MemoryDataSet


def test_sum_dataset_tilesize_1(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16), partition_shape=(1, 8, 16, 16))
    expected = data.sum(axis=(0, 1))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16, 16)
    assert np.allclose(results.intensity.raw_data, expected)


def test_sum_dataset_tilesize_2(lt_ctx):
    data = np.random.choice(a=[0, 1], size=(16, 16, 16, 16)).astype("<u2")
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 16, 16), partition_shape=(1, 8, 16, 16))
    expected = data.sum(axis=(0, 1))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16, 16)
    assert np.allclose(results.intensity.raw_data, expected)


def test_sum_timeseries(lt_ctx):
    """
    sum over the first axis of a 3D dataset
    """
    data = np.random.choice(a=[0, 1], size=(16 * 16, 16, 16)).astype("<u2")
    # FIXME: should tileshape be 3D or 4D here?
    # I think 3D should be fine, as it matches data and partition shape
    dataset = MemoryDataSet(
        data=data,
        effective_shape=(16, 16, 16, 16),
        tileshape=(2, 16, 16),
        partition_shape=(8, 16, 16)
    )

    # only sum over the first axis:
    expected = data.sum(axis=(0,))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16, 16)
    assert np.allclose(results.intensity.raw_data, expected)


def test_sum_spectrum_2d_frames(lt_ctx):
    """
    sum over the first two axes of a 3D dataset
    """
    data = np.random.choice(a=[0, 1], size=(16, 16, 16 * 16)).astype("<u2")
    dataset = MemoryDataSet(
        data=data,
        effective_shape=(16, 16, 16 * 16),
        tileshape=(1, 2, 16 * 16),
        partition_shape=(1, 8, 16 * 16),
        sig_dims=1,
    )

    # sum over the first two axex:
    expected = data.sum(axis=(0, 1))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16 * 16,)
    assert np.allclose(results.intensity.raw_data, expected)


def test_sum_spectrum_linescan(lt_ctx):
    """
    sum over the first axis of a 2D dataset
    """
    data = np.random.choice(a=[0, 1], size=(16 * 16, 16 * 16)).astype("<u2")
    dataset = MemoryDataSet(
        data=data,
        effective_shape=(16 * 16, 16 * 16),
        tileshape=(2, 16 * 16),
        partition_shape=(8, 16 * 16),
        sig_dims=1,
    )

    # only sum over the first axis:
    expected = data.sum(axis=(0,))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16 * 16,)
    assert np.allclose(results.intensity.raw_data, expected)


def test_sum_hyperspectral(lt_ctx):
    # flat navigation dimension to simulate "image stack"-like file formats:
    data = np.random.choice(a=[0, 1], size=(16 * 16, 16, 16, 16)).astype("<u2")
    dataset = MemoryDataSet(
        data=data,
        effective_shape=(16, 16, 16, 16, 16),
        tileshape=(1, 16, 16, 16),
        partition_shape=(8, 16, 16, 16),
        sig_dims=3,
    )

    expected = data.sum(axis=(0,))
    analysis = lt_ctx.create_sum_analysis(dataset=dataset)
    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16, 16, 16)
    assert np.allclose(results.intensity.raw_data, expected)


def test_sum_complex(lt_ctx, ds_complex):
    expected = ds_complex.data.sum(axis=(0, 1))
    analysis = lt_ctx.create_sum_analysis(dataset=ds_complex)
    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16, 16)
    assert np.allclose(results.intensity.raw_data, expected)
