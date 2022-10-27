import numpy as np

from libertem.analysis.sd import SDAnalysis
from libertem.io.dataset.memory import MemoryDataSet


def test_sd_analysis_smoke(lt_ctx):
    data = np.zeros([3*3, 8, 8]).astype(np.float32)
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            num_partitions=2, sig_dims=2)
    analysis = SDAnalysis(dataset=dataset, parameters={})
    lt_ctx.run(analysis)
