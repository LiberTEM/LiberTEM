import os
import math
import mmap
import contextlib

import numpy as np


def _alloc_aligned(size):
    # round up to 4k blocks:
    blocksize = 4096
    blocks = math.ceil(size / blocksize)

    # MAP_SHARED to prevent possible corruption (see open(2))
    return mmap.mmap(-1, blocksize * blocks, mmap.MAP_SHARED)


def empty_aligned(size, dtype):
    dtype = np.dtype(dtype)
    buf = _alloc_aligned(dtype.itemsize * size)
    # _alloc_aligned may give us more memory (for alignment reasons), so crop it off the end:
    npbuf = np.frombuffer(buf, dtype=dtype)[:size]
    return npbuf


@contextlib.contextmanager
def open_direct(path):
    """
    open `path` for reading with O_DIRECT
    """
    fh = os.open(path, os.O_RDONLY | os.O_DIRECT)
    f = open(fh, "rb", buffering=0)
    yield f
    os.close(fh)


def readinto_direct(f, out):
    """
    read from `f` into numpy array `out`, returning a view
    for the read data
    """
    ret = f.readinto(out)
    if ret is None:
        raise IOError("could not readinto()")
    return out[:ret]
