# import functools
import collections

import numba
import numpy as np

from libertem.common.buffers import BufferWrapper

# from libertem.udf import ResultBuffer, map_frames

VariancePart = collections.namedtuple('VariancePart', ['sum_var', 'sum_im', 'N'])


def my_buffer_batch():
    return {
        'batch' : BufferWrapper(
            kind = 'sig', extra_shape = (3,), dtype = 'float32'
            )
    }

def my_frame_fn_batch(frame, batch):

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

def minibatch(partition):
    """
    Given a partition/batch, compute the sum of pixels, the sum of variances, 
    and the number of frames. Return as a collections.namedtuple object
    """
    sum_var = 0
    sum_im = 0
    N = 0

    for frame in partition.get_tiles():
        sum_im += frame.data
        N += 1

    mean = sum_im / N

    for frame in partition.get_tiles():
        sum_var += np.square(mean - frame.data)

    return VariancePart(sum_var = sum_var, sum_im = sum_im, N = N)

def whole(dataset, batchsize = 32):
    """
    Given a subset of images, compute mean and covariance for each pixels.
    """
    #compute mean
    batch = list()

    for partition in dataset.get_partitions():

        minibatch = minibatch(partition)
        batch.append(minibatch)


    # for image in batch:
    #     sum_im += im
    sum_im = np.sum(frame)
    # sum_im = np.sum(batch, axis=0)
    # for im in batch:
    #     sum_im += im
    mean = np.mean(frame)

    #compute sum of variances
    sum_var = 0
    sum_var = np.square(frame - mean)
    for im in batch:
        sum_var += np.square(im.data - mean)

    return VariancePart(sum_var = sum_var, sum_im = sum_im, N = size)

def merge(p0, p1):
    """
    Given two sets of partitions, with mean and sum of variances, 
    compute joint mean and sum of variances using one pass algorithm
    """
    if p0.N == 0:
        return VariancePart(sum_var = p1.sum_var, sum_im = p1.sum_im, N = p1.N)
    N = p0.N + p1.N

    # compute mean for each partitions
    mean_A = (p0.sum_im / p0.N)
    mean_B = (p1.sum_im / p1.N)

    # compute mean for joint samples
    delta = mean_B - mean_A
    mean = mean_A + (p1.N * delta) / (p0.N + p1.N)

    # compute sum of images for joint samples
    sum_im_AB = p0.sum_im + p1.sum_im

    # compute sum of variances for joint samples
    delta_P = mean_B - mean
    sum_var_AB = p0.sum_var + p1.sum_var + (p1.N * delta * delta_P)

    return VariancePart(sum_var=sum_var_AB, sum_im=sum_im_AB, N=N)

# @numba.njit(parallel=True)
def part(dataset):
    """
    Partition the data into given batchsize (default 32) and using minibatch and merge 
    functions, compute mean and standard deviation of images in parallel
    """
    current = None
    batch = list()

    for partition in dataset.get_partitions():

        minib = minibatch(partition)
        batch.append(minib)

    for i in range(len(batch)):

        if i == 0:
            current = batch[i]

        else:
            current = merge(current, batch[i])

    # N = len(data)

    # assert N % batchsize == 0 # number of images is divisible by batchsize

    # for i in range(N // batchsize):

    #     batch = data[i * batchsize:(i + 1) * batchsize]
    #     res = minibatch(batch)

    #     if current is None:
    #         current = res

    #     else:
    #         current = merge(current, res)

    return current


def myvar(p):
    return p.XX / p.N


@numba.njit 
def run_analysis(ctx, dataset, parameters):
    pass
