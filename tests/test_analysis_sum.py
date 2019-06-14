import numpy as np

from utils import MemoryDataSet, _mk_random


def test_sum_dataset_tilesize_1(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype='<u2')
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16), num_partitions=32)
    expected = data.sum(axis=(0, 1))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16, 16)
    assert np.allclose(results.intensity.raw_data, expected)


def test_sum_dataset_tilesize_2(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype='<u2')
    dataset = MemoryDataSet(data=data, tileshape=(8, 16, 16), num_partitions=32)
    expected = data.sum(axis=(0, 1))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16, 16)
    assert np.allclose(results.intensity.raw_data, expected)


def test_sum_endian(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype='>u2')
    dataset = MemoryDataSet(data=data, tileshape=(8, 16, 16), num_partitions=32)
    expected = data.sum(axis=(0, 1))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16, 16)
    assert np.allclose(results.intensity.raw_data, expected)


def test_sum_signed(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype='<i4')
    dataset = MemoryDataSet(data=data, tileshape=(8, 16, 16), num_partitions=32,
                            check_cast=False)
    expected = data.sum(axis=(0, 1))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16, 16)
    assert np.allclose(results.intensity.raw_data, expected)


def test_sum_timeseries(lt_ctx):
    """
    sum over the first axis of a 3D dataset
    """
    data = _mk_random(size=(16 * 16, 16, 16), dtype='<u2')
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, 16, 16),
        num_partitions=32
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
    data = _mk_random(size=(16, 16, 16 * 16), dtype='<u2')
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, 16 * 16),
        num_partitions=32,
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
    data = _mk_random(size=(16 * 16, 16 * 16), dtype='<u2')
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, 16 * 16),
        num_partitions=32,
        sig_dims=1,
    )

    # only sum over the first axis:
    expected = data.sum(axis=(0,))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16 * 16,)
    assert np.allclose(results.intensity.raw_data, expected)


def test_sum_hyperspectral(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16, 16), dtype='<u2')
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16, 16),
        num_partitions=32,
        sig_dims=3,
    )

    expected = data.sum(axis=(0, 1))
    analysis = lt_ctx.create_sum_analysis(dataset=dataset)
    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16, 16, 16)
    assert np.allclose(results.intensity.raw_data, expected)


def test_sum_complex(lt_ctx, ds_complex):
    expected = ds_complex.data.sum(axis=(0, 1))
    analysis = lt_ctx.create_sum_analysis(dataset=ds_complex)
    results = lt_ctx.run(analysis)

    assert results.intensity.raw_data.shape == (16, 16)
    assert np.allclose(results.intensity_complex.raw_data, expected)
