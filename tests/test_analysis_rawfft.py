import numpy as np

from libertem.analysis.rawfft import PickFFTFrameAnalysis
from libertem.io.dataset.memory import MemoryDataSet


def test_pick_fft_defaults(lt_ctx):
    data = np.zeros([3*3, 8, 8]).astype(np.float32)
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            num_partitions=2, sig_dims=2)
    analysis = PickFFTFrameAnalysis(dataset=dataset, parameters={
        'x': 1,
    })
    res = lt_ctx.run(analysis)


def test_pick_fft_masked(lt_ctx):
    data = np.zeros([3*3, 8, 8]).astype(np.float32)
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            num_partitions=2, sig_dims=2)
    analysis = PickFFTFrameAnalysis(dataset=dataset, parameters={
        'x': 1,
        'real_rad': 1,
        'real_centerx': 1,
        'real_centery': 1,
    })
    res = lt_ctx.run(analysis)
