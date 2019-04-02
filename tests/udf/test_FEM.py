import numpy as np

import libertem.udf.FEM as FEM

from utils import MemoryDataSet


def test_smoke(lt_ctx):
    """
    just check if the analysis runs without throwing exceptions:
    """
    data = np.zeros([3*3, 8, 8]).astype(np.float32)
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            partition_shape=(4, 8, 8), sig_dims=2)
    result = FEM.run_fem(ctx=lt_ctx, dataset=dataset, center=(1, 1), rad_in=0, rad_out=1)
    assert np.allclose(result['intensity'].data, np.zeros(data.shape[0]))


def test_smoke2(lt_ctx):
    """
    just check if the analysis runs without throwing exceptions:
    """
    data = np.zeros([3*3, 3, 3]).astype(np.float32) + [[0, 1, 0], [3, 0, 3], [0, 1, 0]]
    data[8] = data[8] + [[0, 0, 0], [0, 0, 0], [0, 0, 9]]
    dataset = MemoryDataSet(data=data, tileshape=(1, 3, 3),
                            partition_shape=(4, 3, 3), sig_dims=2)
    result = FEM.run_fem(ctx=lt_ctx, dataset=dataset, center=(1, 1), rad_in=0, rad_out=1)
    assert np.allclose(result['intensity'].data, np.ones(data.shape[0]))