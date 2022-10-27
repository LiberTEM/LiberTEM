import numpy as np

from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random

from libertem import masks
from libertem.analysis.sum import SumAnalysis


def test_sum_dataset_tilesize_1(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype='<u2')
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16), num_partitions=32)
    expected = data.sum(axis=(0, 1))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results['intensity'].raw_data.shape == (16, 16)
    assert np.allclose(results['intensity'].raw_data, expected)
    assert np.allclose(results['intensity_lin'].raw_data, expected)


def test_sum_dataset_tilesize_2(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype='<u2')
    dataset = MemoryDataSet(data=data, tileshape=(8, 16, 16), num_partitions=32)
    expected = data.sum(axis=(0, 1))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results['intensity'].raw_data.shape == (16, 16)
    assert np.allclose(results['intensity'].raw_data, expected)
    assert np.allclose(results['intensity_lin'].raw_data, expected)


def test_sum_endian(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype='>u2')
    dataset = MemoryDataSet(data=data, tileshape=(8, 16, 16), num_partitions=32)
    expected = data.sum(axis=(0, 1))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results['intensity'].raw_data.shape == (16, 16)
    assert np.allclose(results['intensity'].raw_data, expected)
    assert np.allclose(results['intensity_lin'].raw_data, expected)


def test_sum_signed(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype='<i4')
    dataset = MemoryDataSet(data=data, tileshape=(8, 16, 16), num_partitions=32,
                            check_cast=False)
    expected = data.sum(axis=(0, 1))

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)

    results = lt_ctx.run(analysis)

    assert results['intensity'].raw_data.shape == (16, 16)
    assert np.allclose(results['intensity'].raw_data, expected)
    assert np.allclose(results['intensity_lin'].raw_data, expected)


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

    assert results['intensity'].raw_data.shape == (16, 16)
    assert np.allclose(results['intensity'].raw_data, expected)
    assert np.allclose(results['intensity_lin'].raw_data, expected)


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

    assert results['intensity'].raw_data.shape == (16 * 16,)
    assert np.allclose(results['intensity'].raw_data, expected)
    assert np.allclose(results['intensity_lin'].raw_data, expected)


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

    assert results['intensity'].raw_data.shape == (16 * 16,)
    assert np.allclose(results['intensity'].raw_data, expected)
    assert np.allclose(results['intensity_lin'].raw_data, expected)


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

    assert results['intensity'].raw_data.shape == (16, 16, 16)
    assert np.allclose(results['intensity'].raw_data, expected)
    assert np.allclose(results['intensity_lin'].raw_data, expected)


def test_sum_complex(lt_ctx, ds_complex):
    expected = ds_complex.data.sum(axis=(0, 1))
    analysis = lt_ctx.create_sum_analysis(dataset=ds_complex)
    results = lt_ctx.run(analysis)

    assert ds_complex.data.dtype.kind == 'c'
    assert results['intensity_complex'].raw_data.dtype.kind == 'c'

    assert results['intensity'].raw_data.shape == (16, 16)
    assert np.allclose(results['intensity_complex'].raw_data, expected)
    assert np.allclose(results['intensity_lin'].raw_data, np.abs(expected))
    assert np.allclose(results['intensity'].raw_data, np.abs(expected))


def test_sum_with_roi(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype='<u2')
    dataset = MemoryDataSet(data=data, tileshape=(2, 16, 16), num_partitions=32)

    roi = {
        "shape": "disk",
        "cx": 5,
        "cy": 6,
        "r": 7,
    }
    analysis = SumAnalysis(dataset=dataset, parameters={
        "roi": roi,
    })

    results = lt_ctx.run(analysis)

    mask = masks.circular(roi["cx"], roi["cy"], 16, 16, roi["r"])
    assert mask.shape == (16, 16)
    assert mask[0, 0] == 0
    assert mask[6, 5] == 1
    assert mask.dtype == bool

    # applying the mask flattens the first two dimensions, so we
    # only sum over axis 0 here:
    expected = data[mask, ...].sum(axis=(0,))

    assert expected.shape == (16, 16)
    assert results['intensity'].raw_data.shape == (16, 16)

    # is not equal to results without mask:
    assert not np.allclose(results['intensity'].raw_data, data.sum(axis=(0, 1)))
    # ... but rather like `expected`:
    assert np.allclose(results['intensity'].raw_data, expected)
    assert np.allclose(results['intensity_lin'].raw_data, expected)


def test_sum_zero_roi(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype='<u2')
    dataset = MemoryDataSet(data=data, tileshape=(2, 16, 16), num_partitions=32)

    roi = {
        "shape": "disk",
        "cx": -1,
        "cy": -1,
        "r": 0,
    }
    analysis = SumAnalysis(dataset=dataset, parameters={
        "roi": roi,
    })

    results = lt_ctx.run(analysis)

    mask = masks.circular(roi["cx"], roi["cy"], 16, 16, roi["r"])
    assert mask.shape == (16, 16)
    assert np.count_nonzero(mask) == 0
    assert mask.dtype == bool

    # applying the mask flattens the first two dimensions, so we
    # only sum over axis 0 here:
    expected = data[mask, ...].sum(axis=(0,))

    assert expected.shape == (16, 16)
    assert results['intensity'].raw_data.shape == (16, 16)

    # is not equal to results without mask:
    assert not np.allclose(results['intensity'].raw_data, data.sum(axis=(0, 1)))
    # ... but rather like `expected`:
    assert np.allclose(results['intensity'].raw_data, expected)
    assert np.allclose(results['intensity_lin'].raw_data, expected)


def test_sum_with_crop_frames(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(7, 8, 8),
                            num_partitions=2, sig_dims=2)

    analysis = lt_ctx.create_sum_analysis(dataset=dataset)
    res = lt_ctx.run(analysis)
    print(data.shape, res.intensity.raw_data.shape)
    assert np.allclose(res.intensity.raw_data, np.sum(data, axis=(0, 1)))
