import logging
from contextlib import contextmanager

import threadpoolctl
try:
    import pyfftw
except ImportError:
    pyfftw = None

import numba


log = logging.getLogger(__name__)


if pyfftw:
    @contextmanager
    def set_fftw_threads(n):
        pyfftw_threads = pyfftw.config.NUM_THREADS
        try:
            pyfftw.config.NUM_THREADS = n
            yield
        finally:
            pyfftw.config.NUM_THREADS = pyfftw_threads
else:
    @contextmanager
    def set_fftw_threads(n):
        yield


@contextmanager
def set_torch_threads(n):
    try:
        import torch
    except ImportError:
        yield
        return
    torch_threads = torch.get_num_threads()
    # See also https://pytorch.org/docs/stable/torch.html#parallelism
    # At the time of writing the difference between threads and interop threads
    # wasn't clear from the documentation. However, changing the
    # interop_threads on runtime
    # caused errors, so it is commented out here.
    # torch_interop_threads = torch.get_num_interop_threads()
    try:
        torch.set_num_threads(n)
        # torch.set_num_interop_threads(n)
        yield
    finally:
        torch.set_num_threads(torch_threads)
        # torch.set_num_interop_threads(torch_interop_threads)


@contextmanager
def set_numba_threads(n):
    numba_threads = numba.get_num_threads()
    try:
        numba.set_num_threads(n)
        yield
    finally:
        numba.set_num_threads(numba_threads)


@contextmanager
def set_num_threads(n):
    with threadpoolctl.threadpool_limits(n), set_fftw_threads(n),\
            set_torch_threads(n), set_numba_threads(n):
        yield
