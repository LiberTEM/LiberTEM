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
        real_center=(1,1), real_rad=1)
    assert np.allclose(result['intensity'].data, np.zeros(data.shape[0]))

