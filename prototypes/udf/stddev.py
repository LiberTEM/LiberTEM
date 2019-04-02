# import functools
import collections

import numba
import numpy as np

# from libertem.udf import ResultBuffer, map_frames

try:
    import pyfftw
    fft = pyfftw.interfaces.numpy_fft
    pyfftw.interfaces.cache.enable()
    zeros = pyfftw.zeros_aligned
except ImportError:
    fft = np.fft
    zeros = np.zeros


VariancePart = collections.namedtuple('VariancePart', ['XX', 'X', 'N'])


@numba.njit
def minibatch(batch):
    size = len(batch)
    X = 0
    for x in batch:
        X += x
    mean = X / size
    XX = 0
    for x in batch:
        XX += (x - mean) ** 2
    return VariancePart(XX=XX, X=X, N=size)


@numba.njit
def part(data):
    batchsize = 32
    current = None
    N = len(data)

    assert N % batchsize == 0
    for i in range(N // batchsize):
        batch = data[i * batchsize:(i + 1) * batchsize]
        res = minibatch(batch)
        if current is None:
            current = res
        else:
            current = merge(current, res)
    return current


@numba.njit
def merge(p0, p1):
    N = p0.N + p1.N
    mean_A = (p0.X / p0.N)
    mean_B = (p1.X / p1.N)
    delta = mean_B - mean_A
    mean = mean_A + (p1.N * delta) / (p0.N + p1.N)
    delta_P = mean_B - mean
    XX_AB = p0.XX + p1.XX + (p1.N * delta * delta_P)
    X_AB = N * mean
    return VariancePart(XX=XX_AB, X=X_AB, N=N)


def myvar(p):
    return p.XX / p.N


def run_analysis(ctx, dataset, parameters):
    pass
