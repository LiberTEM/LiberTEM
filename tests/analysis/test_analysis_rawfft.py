import numpy as np
from numpy.testing import assert_allclose

from libertem.analysis.rawfft import PickFFTFrameAnalysis
from libertem.io.dataset.memory import MemoryDataSet
from libertem.masks import _make_circular_mask

from utils import _mk_random


def test_pick_fft_defaults(lt_ctx):
    data = _mk_random([3*3, 8, 8], dtype=np.float32)
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            num_partitions=2, sig_dims=2)
    analysis = PickFFTFrameAnalysis(dataset=dataset, parameters={
        'x': 1,
    })
    res = lt_ctx.run(analysis)
    fft_data = np.fft.fftshift(abs(np.fft.fft2(data[1])))

    assert np.allclose(res.intensity.raw_data, fft_data)


def test_pick_fft_masked(lt_ctx):
    data = _mk_random([3*3, 8, 8], dtype=np.float32)
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            num_partitions=2, sig_dims=2)
    analysis = PickFFTFrameAnalysis(dataset=dataset, parameters={
        'x': 1,
        'real_rad': 1,
        'real_centerx': 1,
        'real_centery': 1,
    })
    real_mask = np.invert(_make_circular_mask(
        centerX=1, centerY=1, imageSizeX=8, imageSizeY=8, radius=1
    ))
    fft_data = np.fft.fftshift(abs(np.fft.fft2(data[1]*real_mask)))
    res = lt_ctx.run(analysis)

    assert_allclose(res.intensity.raw_data, fft_data, rtol=1e-6, atol=1e-6, equal_nan=True)
