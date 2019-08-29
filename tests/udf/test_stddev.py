import numpy as np

from libertem.udf.stddev import run_stddev
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


def test_stddev(lt_ctx):
    """
    Test variance, standard deviation, sum of frames, and mean computation
    implemented in udf/stddev.py

    Parameters
    ----------
    lt_ctx
        Context class for loading dataset and creating jobs on them
    """
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    res = run_stddev(lt_ctx, dataset)

    assert 'sum_frame' in res
    assert 'num_frame' in res
    assert 'var' in res
    assert 'mean' in res
    assert 'std' in res

    N = data.shape[2] * data.shape[3]
    assert res['num_frame'] == N  # check the total number of frames

    assert np.allclose(res['sum_frame'], np.sum(data, axis=(0, 1)))  # check sum of frames

    assert np.allclose(res['mean'], np.mean(data, axis=(0, 1)))  # check mean

    var = np.var(data, axis=(0, 1))
    assert np.allclose(var, res['var'])  # check variance

    std = np.std(data, axis=(0, 1))
    assert np.allclose(std, res['std'])  # check standard deviation
