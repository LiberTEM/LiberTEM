import numpy as np
import numba

from libertem.common.numba import cached_njit


def fn(a, b):
    return a ** b


cached_fn = cached_njit(fastmath=True)(fn)
numba_fn = numba.njit(fastmath=True, cache=True)(fn)

cached_fn(np.float32(1.2345), 7)  # call 1
numba_fn(np.float32(1.2345), 7)  # call 2
