import numpy as np

from libertem.analysis.sumfft import SumfftAnalysis
from libertem.io.dataset.memory import MemoryDataSet


def test_sum_fft_analysis_defaults(lt_ctx):
    data = np.zeros([3*3, 8, 8]).astype(np.float32)
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            num_partitions=2, sig_dims=2)
    analysis = SumfftAnalysis(dataset=dataset, parameters={})
    lt_ctx.run(analysis)


def test_sum_fft_analysis_masked(lt_ctx):
    data = np.zeros([3*3, 8, 8]).astype(np.float32)
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            num_partitions=2, sig_dims=2)
    analysis = SumfftAnalysis(dataset=dataset, parameters={
        'real_rad': 1,
        'real_centerx': 1,
        'real_centery': 1,
    })
    lt_ctx.run(analysis)
