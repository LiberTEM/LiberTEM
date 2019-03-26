# import functools
import collections

import numba
import numpy as np

from libertem.common.buffers import BufferWrapper

# from libertem.udf import ResultBuffer, map_frames

VariancePart = collections.namedtuple('VariancePart', ['sum_var', 'sum_im', 'N'])

def make_stddev_buffer():
    """ 
    Buffer to store the resulting standard deviation image 
    """
    return {
        'stddev' : BufferWrapper(
            kind = 'sig', dtype = 'float32'
            )
    }

def minibatch(batch):
    """
    Given a subset of images, compute mean and covariance for each pixels.
    """
    size = len(batch)
    sum_im = 0

    #compute mean 
    for im in batch:
        sum_im += im
    mean = sum_im / size

    #compute sum of variances
    sum_var = 0
    for im in batch:
        sum_var += np.square(im - mean)

    return VariancePart(sum_var = sum_var, sum_im = sum_im, N = size)

def merge(p0, p1):
    """
    Given two sets of partitions, with mean and sum of variances, 
    compute joint mean and sum of variances using one pass algorithm
    """
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

def part(data, batchsize = 32):
    """
    Partition the data into given batchsize (default 32) and using minibatch and merge 
    functions, compute mean and standard deviation of images in parallel
    """
    current = None
    N = len(data)

    assert N % batchsize == 0 # number of images is divisible by batchsize

    for i in range(N // batchsize):

        batch = data[i * batchsize:(i + 1) * batchsize]
        res = minibatch(batch)

        if current is None:
            current = res

        else:
            current = merge(current, res)

    return current


def myvar(p):
    return p.XX / p.N


@numba.njit 
def run_analysis(ctx, dataset, parameters):
    pass
