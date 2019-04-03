import numpy as np
import pytest

from libertem.common.buffers import BufferWrapper
from libertem.api import Context
from libertem.udf.stddev import merge, batch_merge, compute_batch, batch_buffer, run_stddev
from utils import MemoryDataSet, _mk_random


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
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16),
                            partition_shape=(4, 4, 16, 16), sig_dims=2)

    res = run_stddev(lt_ctx, dataset)

    assert 'sum_frame' in res
    assert 'num_frame' in res
    assert 'var' in res
    assert 'mean' in res
    assert 'std' in res 

    N = data.shape[2] * data.shape[3]
    assert res['num_frame'].data == N # check the total number of frames

    assert np.allclose(res['sum_frame'].data, np.sum(data, axis=(0, 1))) # check sum of frames

    assert np.allclose(res['mean'].data, np.mean(data, axis=(0, 1))) # check mean

    var = np.var(data, axis=(0, 1))
    assert np.allclose(var, res['var'].data) # check variance

    std = np.std(data, axis=(0, 1))
    assert np.allclose(std, res['std'].data) # check standard deviation