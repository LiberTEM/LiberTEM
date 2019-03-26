import collections 

import numpy as np

from libertem.common.buffers import BufferWrapper
from libertem.api import Context
from libertem.udf.stddev import minibatch, merge, part
from utils import MemoryDataSet, _mk_random

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
    Test minibatch function from libertem/udf/stddev.py 
    """
    batch = [np.random.rand(2, 3) for i in range(batchsize)]
    compute_batch = minibatch(batch)

    sum_var, sum_im, N = compute_batch.sum_var, compute_batch.sum_im, compute_batch.N

    assert N == len(batch) # check number of observations
    assert np.allclose([sum_im], [sum(batch)]) # check sum of image pixels 

    # check sum of variances 
    var = np.var(batch, axis=0) * batchsize
    assert np.allclose([var], [sum_var]) 

def test_merge(batchsize = 64):
    """
    Test merge function from libertem/udf/stddev.py
    """
    batch = [np.random.rand(2, 3) for i in range(batchsize)]

    obs0 = np.random.randint(5, 45)
    obs1 = batchsize - obs0

    batch0 = batch[:obs0]
    batch1 = batch[obs0:]

    sum_im0 = sum(batch0)
    sum_im1 = sum(batch1)

    sum_var0 = np.var(batch0, axis=0) * obs0
    sum_var1 = np.var(batch1, axis=0) * obs1

    VariancePart = collections.namedtuple('VariancePart', ['sum_var', 'sum_im', 'N'])

    p0 = VariancePart(sum_var = sum_var0, sum_im = sum_im0, N = obs0)
    p1 = VariancePart(sum_var = sum_var1, sum_im = sum_im1, N = obs1)

    compute_merge = merge(p0, p1)
    sum_var, sum_im, N = compute_merge.sum_var, compute_merge.sum_im, compute_merge.N

    assert N == obs0 + obs1 

    # check sum of pixels 
    sum_ims = sum(batch)
    assert np.allclose([sum_ims], [sum_im])

    # check sum of variances 
    var = np.var(batch, axis = 0) * batchsize
    assert np.allclose([var], [sum_var])
