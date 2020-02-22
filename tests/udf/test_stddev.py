import pytest
import numpy as np
import numba

from libertem.udf.stddev import run_stddev, process_tile, merge
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


@pytest.mark.with_numba
@pytest.mark.parametrize(
    "use_roi", [True, False]
)
def test_stddev(lt_ctx, use_roi):
    """
    Test variance, standard deviation, sum of frames, and mean computation
    implemented in udf/stddev.py

    Parameters
    ----------
    lt_ctx
        Context class for loading dataset and creating jobs on them
    """
    data = _mk_random(size=(16, 17, 18, 19), dtype="float32")
    # FIXME the tiling in signal dimension can only be tested once MemoryDataSet
    # actually supports it
    dataset = MemoryDataSet(data=data, tileshape=(3, 2, 16),
                            num_partitions=8, sig_dims=2)
    if use_roi:
        roi = np.random.choice([True, False], size=dataset.shape.nav)
        res = run_stddev(lt_ctx, dataset, roi=roi)
    else:
        roi = np.ones(dataset.shape.nav, dtype=bool)
        res = run_stddev(lt_ctx, dataset)

    assert 'sum_frame' in res
    assert 'num_frame' in res
    assert 'var' in res
    assert 'mean' in res
    assert 'std' in res

    N = np.count_nonzero(roi)
    assert res['num_frame'] == N  # check the total number of frames

    print(res['sum_frame'])
    print(np.sum(data[roi], axis=0))
    print(res['sum_frame'] - np.sum(data[roi], axis=0))
    assert np.allclose(res['sum_frame'], np.sum(data[roi], axis=0))  # check sum of frames

    assert np.allclose(res['mean'], np.mean(data[roi], axis=0))  # check mean

    var = np.var(data[roi], axis=0)
    assert np.allclose(var, res['var'])  # check variance

    std = np.std(data[roi], axis=0)
    assert np.allclose(std, res['std'])  # check standard deviation


@numba.njit
def _stability_workhorse(data):
    s = np.zeros(1, dtype=np.float32)
    varsum = np.zeros(1, dtype=np.float32)
    N = 0
    partitions = data.shape[0]
    tiles = data.shape[1]
    frames = data.shape[2]
    for partition in range(partitions):
        partition_sum = np.zeros(1, dtype=np.float32)
        partition_varsum = np.zeros(1, dtype=np.float32)
        partition_N = 0
        for tile in range(tiles):
            if partition_N == 0:
                partition_sum[:] = np.sum(data[partition, tile])
                partition_varsum[:] = np.var(data[partition, tile]) * frames
                partition_N = frames
            else:
                partition_N = process_tile(
                    tile=data[partition, tile],
                    N0=partition_N,
                    sum_inout=partition_sum,
                    var_inout=partition_varsum
                )
        N = merge(
            dest_N=N,
            dest_sum=s,
            dest_varsum=varsum,
            src_N=partition_N,
            src_sum=partition_sum,
            src_varsum=partition_varsum
        )
    return N, s, varsum


# This shouldn't be @pytest.mark.numba
# since the calculation will take forever
# with JIT disabled
def test_stability(lt_ctx):
    """
    Test variance, standard deviation, sum of frames, and mean computation
    implemented in udf/stddev.py

    Parameters
    ----------
    lt_ctx
        Context class for loading dataset and creating jobs on them
    """
    data = _mk_random(size=(1024, 1024, 8, 1), dtype="float32")

    N, s, varsum = _stability_workhorse(data)

    assert N == np.prod(data.shape)
    assert np.allclose(data.sum(), s)
    assert np.allclose(data.var(ddof=N-1), varsum)
