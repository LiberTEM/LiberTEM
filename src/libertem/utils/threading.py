import logging
from contextlib import contextmanager
import os
import warnings

# NOTE: most imports are performed locally in the functions, to
# make sure these functions are usable as early as possible in the
# start-up of libertem (i.e. before numpy/numba/... are imported).
# In particular, that makes sure the environment variables can be set
# with set_num_threads_env() before the libraries are loaded.


log = logging.getLogger(__name__)


__has_pyfftw = None
__has_pytorch = None


@contextmanager
def set_fftw_threads(n):
    global __has_pyfftw
    if __has_pyfftw is None:
        try:
            import pyfftw
            __has_pyfftw = True
        except ImportError:
            __has_pyfftw = False
            yield
            return

    if __has_pyfftw:
        import pyfftw  # noqa:F811
        pyfftw_threads = pyfftw.config.NUM_THREADS
        try:
            pyfftw.config.NUM_THREADS = n
            yield
        finally:
            pyfftw.config.NUM_THREADS = pyfftw_threads
    else:
        yield


@contextmanager
def set_torch_threads(n):
    global __has_pytorch
    if __has_pytorch is None:
        try:
            import torch
            __has_pytorch = True
        except ImportError:
            __has_pytorch = False
            yield
            return

    if __has_pytorch:
        import torch  # noqa:F811
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
    else:
        yield


@contextmanager
def set_numba_threads(n):
    import numba
    from numba.core.config import NUMBA_NUM_THREADS
    numba_threads = numba.get_num_threads()
    try:
        if n > NUMBA_NUM_THREADS:
            warnings.warn(
                f"Attempting to set threads to {n}, which is larger than "
                f"NUMBA_NUM_THREADS={NUMBA_NUM_THREADS}. "
                f"Setting to allowed maximum NUMBA_NUM_THREADS instead."
            )
        n = min(n, NUMBA_NUM_THREADS)

        numba.set_num_threads(n)
        yield
    finally:
        numba.set_num_threads(numba_threads)


class ThreadpoolWrapper:
    def __init__(self, repeats=2):
        import threadpoolctl
        self._info = threadpoolctl.threadpool_info()
        self._controller = threadpoolctl.ThreadpoolController()
        self._stable = 0
        self.repeats = repeats

    def check_update(self):
        import threadpoolctl
        new_info = threadpoolctl.threadpool_info()
        if new_info != self._info:
            self._stable = 0
            self._controller = threadpoolctl.ThreadpoolController()
            self._info = new_info
        else:
            self._stable += 1

    # Currently not used due to executor overhead
    # def reset(self):
    #     self ._stable = 0

    def __call__(self, limits):
        if self._stable < self.repeats:
            self.check_update()
        return self._controller.limit(limits=limits)


# Make sure we don't load threadpoolctl when importing,
# just to be sure no C library gets loaded!
__threadpool_wrapper = None


# Currently not used due to executor overhead
# def reset_module_cache():
#     if __threadpool_wrapper is not None:
#         __threadpool_wrapper.reset()


@contextmanager
def set_num_threads(n):
    # Make sure modules that use BLAS are loaded
    import scipy  # noqa: F401
    import numpy as np  # noqa: F401
    global __threadpool_wrapper
    if __threadpool_wrapper is None:
        __threadpool_wrapper = ThreadpoolWrapper()
    # We use __threadpool_wrapper last so that it can cover
    # libraries that the other ones load
    with set_fftw_threads(n):
        with set_torch_threads(n):
            with set_numba_threads(n):
                with __threadpool_wrapper(n):
                    yield


@contextmanager
def set_num_threads_env(n=1, set_numba=None):
    """Set the maximum number of threads via environment variables.
    Currently sets variables for MKL, OMP, OPENBLAS and NUMBA.

    This is needed if you want to limit the number of threads that
    are created at startup of those libraries, because the other
    methods usually only limit the number of threads that are _used_,
    not the number of threads that are created.

    Parameters
    ----------
    n : int
        The maximum number of threads
    set_numba : bool, optional
        Set the :code:`'NUMBA_NUM_THREADS'` environment variable. If None, determine if
        Numba has been initialized and only set it if not.
        Numba may throw a :class:`RuntimeError` if the environment is
        altered after the number of threads has been set already.
    """
    if set_numba is None:
        from numba.np.ufunc import parallel
        set_numba = not getattr(parallel, "_is_initialized", True)
    try:
        env_keys = [
            "MKL_NUM_THREADS", "OMP_NUM_THREADS",
            "OPENBLAS_NUM_THREADS"
        ]
        if set_numba:
            env_keys.append("NUMBA_NUM_THREADS")
        old_env = {
            k: os.environ[k]
            for k in env_keys
            if k in os.environ
        }
        os.environ.update({
            k: str(n)
            for k in env_keys
        })
        yield
    finally:
        os.environ.update(old_env)
        for key in env_keys:
            if key not in old_env:
                del os.environ[key]
