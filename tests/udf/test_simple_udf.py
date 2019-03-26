import numpy as np
from libertem.common.buffers import BufferWrapper
from libertem.api import Context
from libertem.udf.stddev import minibatch, merge, part
import collections 

if __name__ == '__main__':
    if  __package__ is None:
        from os import sys, path
        sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
        from utils import MemoryDataSet, _mk_random
    else:
        from ..utils import MemoryDataSet, _mk_random

def test_sum_frames(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16),
                            partition_shape=(4, 4, 16, 16), sig_dims=2)

    def my_buffers():
        return {
            'pixelsum': BufferWrapper(
                kind="nav", dtype="float32"
            )
        }

    def my_frame_fn(frame, pixelsum):
        pixelsum[:] = np.sum(frame)

    res = lt_ctx.run_udf(
        dataset=dataset,
        fn=my_frame_fn,
        make_buffers=my_buffers,
    )
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(2, 3)))


def test_3d_ds(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            partition_shape=(4, 16, 16), sig_dims=2)

    def my_buffers():
        return {
            'pixelsum': BufferWrapper(
                kind="nav", dtype="float32"
            )
        }

    def my_frame_fn(frame, pixelsum):
        pixelsum[:] = np.sum(frame)

    res = lt_ctx.run_udf(
        dataset=dataset,
        fn=my_frame_fn,
        make_buffers=my_buffers,
    )
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(1, 2)))

def test_minibatch(batchsize = 32):
    """
    Test minibatch function from stddev.py 
    """
    batch = [np.random.rand(2, 3) for i in range(batchsize)]
    compute_batch = minibatch(batch)

    sum_var, sum_im, N = compute_batch.sum_var, compute_batch.sum_im, compute_batch.N

    assert N == len(batch) # check number of observations
    assert np.allclose([sum_im], [sum(batch)]) # check sum of image pixels 

    # check sum of variances 
    mean = sum(batch)/len(batch)
    var = sum([np.square(mean - im) for im in batch])
    assert np.allclose([var], [sum_var]) 
