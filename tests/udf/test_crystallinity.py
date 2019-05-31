import numpy as np

import libertem.udf.crystallinity as crystal

from utils import MemoryDataSet


def test_smoke(lt_ctx):
    """
    just check if the analysis runs without throwing exceptions:
    """
    data = np.zeros([3*3, 8, 8]).astype(np.float32)
    dataset = MemoryDataSet(data=data, tileshape=(1, 8, 8),
                            num_partitions=2, sig_dims=2)
    result = crystal.run_analysis_crystall(
        ctx=lt_ctx, dataset=dataset, rad_in=0, rad_out=1,
        real_center=(1, 1), real_rad=1)
    assert np.allclose(result['intensity'].data, np.zeros(data.shape[0]))


def test_smoke2(lt_ctx):
    """
    just check if the analysis runs without throwing exceptions:
    """
    data = np.zeros([3*3, 5, 5]).astype(np.float32)
    data[:, 2, 2] = 7
    data[0:3, 0, 0] = 2
    data[0:3, 4, 4] = 2
    data[3:6, 2, 0] = 1
    data[3:6, 2, 4] = 1
    dataset = MemoryDataSet(data=data, tileshape=(1, 5, 5),
                            num_partitions=3, sig_dims=2)
    result = crystal.run_analysis_crystall(ctx=lt_ctx, dataset=dataset, rad_in=0, rad_out=3, 
    real_center=(2, 2), real_rad=0
    )
    assert np.allclose(result['intensity'].data[6:9], np.zeros([3]))
    assert (result['intensity'].data[0:3] > np.zeros([3])).all()
    assert (result['intensity'].data[3:6] > np.zeros([3])).all()
    assert (result['intensity'].data[0:3] > result['intensity'].data[3:6]).all()
