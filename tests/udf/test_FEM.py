import numpy as np

import libertem.udf.FEM as FEM
from libertem.analysis.fem import FEMAnalysis

from libertem.io.dataset.memory import MemoryDataSet


def test_smoke(lt_ctx):
    """
    just check if the analysis runs without throwing exceptions:
    """
    data = np.zeros([3*3, 8, 8]).astype(np.float32)
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            num_partitions=2, sig_dims=2)
    result = FEM.run_fem(ctx=lt_ctx, dataset=dataset, center=(1, 1), rad_in=0, rad_out=1)
    assert np.allclose(result['intensity'].data, np.zeros(data.shape[0]))


def test_different_size(lt_ctx):
    """
    just check if the analysis runs without throwing exceptions:
    """
    data = np.zeros([3*3, 3, 3]).astype(np.float32) + [[0, 1, 0], [3, 0, 3], [0, 1, 0]]
    data[8] = data[8] + [[0, 0, 0], [0, 0, 0], [0, 0, 9]]
    dataset = MemoryDataSet(data=data, tileshape=(1, 3, 3),
                            num_partitions=2, sig_dims=2)
    result = FEM.run_fem(ctx=lt_ctx, dataset=dataset, center=(1, 1), rad_in=0, rad_out=1)
    assert np.allclose(result['intensity'].data, np.ones(data.shape[0]))


def test_fem_analysis(lt_ctx):
    data = np.zeros([3*3, 8, 8]).astype(np.float32)
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            num_partitions=2, sig_dims=2)
    fem_analysis = FEMAnalysis(parameters={
        'cx': 1,
        'cy': 1,
        'ri': 0,
        'ro': 1,
    }, dataset=dataset)

    result = lt_ctx.run(fem_analysis)
    assert np.allclose(result['intensity'].raw_data, np.zeros(data.shape[0]))
