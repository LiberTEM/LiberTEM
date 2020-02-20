import pytest
import numpy as np

from libertem.udf.stddev import run_stddev
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


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

    assert np.allclose(res['sum_frame'], np.sum(data[roi], axis=0))  # check sum of frames

    assert np.allclose(res['mean'], np.mean(data[roi], axis=0))  # check mean

    var = np.var(data[roi], axis=0)
    assert np.allclose(var, res['var'])  # check variance

    std = np.std(data[roi], axis=0)
    assert np.allclose(std, res['std'])  # check standard deviation
