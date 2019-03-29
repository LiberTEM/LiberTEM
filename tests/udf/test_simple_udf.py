import collections

import numpy as np

from libertem.common.buffers import BufferWrapper
from libertem.api import Context
from libertem.udf.stddev import merge
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

VariancePart = collections.namedtuple('VariancePart', ['sum_var', 'sum_im', 'N'])

def my_buffers():
    return {
        'batch' : BufferWrapper(
            kind = 'sig', extra_shape = (3,), dtype = 'float32'
            )
    }

def my_frame_fn(frame, batch):

    if batch[:, :, 2][0][0] == 0:
        batch[:, :, 0] = 0

    else:
        
        p0 = VariancePart(sum_var = batch[:, :, 0], sum_im = batch[:, :, 1], N = batch[:, :, 2][0][0])
        p1 = VariancePart(sum_var = 0, sum_im = frame, N = 1)
        compute_merge = merge(p0, p1)

        sum_var, sum_im, N = compute_merge.sum_var, compute_merge.sum_im, compute_merge.N

        batch[:, :, 0] = sum_var

    batch[:, :, 1] += frame
    batch[:, :, 2] += 1

def stddev_merge(dest, src):

    p0 = VariancePart(sum_var = dest['batch'][:, :, 0], sum_im = dest['batch'][:, :, 1], N = dest['batch'][:, :, 2][0][0])
    p1 = VariancePart(sum_var = src['batch'][:, :, 0], sum_im = src['batch'][:, :, 1], N = src['batch'][:, :, 2][0][0])
    compute_merge = merge(p0, p1)

    sum_var, sum_im, N = compute_merge.sum_var, compute_merge.sum_im, compute_merge.N
    dest['batch'][:, :, 0] = sum_var
    dest['batch'][:, :, 1] = sum_im
    dest['batch'][:, :, 2] = N

def test_minibatch(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16),
                            partition_shape=(4, 4, 16, 16), sig_dims=2)

    res = lt_ctx.run_udf(
        dataset=dataset,
        fn=my_frame_fn,
        make_buffers=my_buffers,
        merge=stddev_merge,
    )

    assert 'batch' in res

    N = data.shape[2] * data.shape[3]
    assert res['batch'].data[:, :, 2][0][0] == N # check the total number of frames 

    assert np.allclose(res['batch'].data[:, :, 1], np.sum(data, axis = (0, 1))) # check sum of frames

    sum_var = np.var(data, axis=(0, 1))
    assert np.allclose(sum_var, res['batch'].data[:, :, 0]/N) # check sum of variances
