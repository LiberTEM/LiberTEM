import logging
from contextlib import contextmanager
import os
import warnings

# NOTE: most imports are performed locally in the functions, to
# make sure these functions are usable as early as possible in the
# start-up of libertem (i.e. before numpy/numba/... are imported)


log = logging.getLogger(__name__)


@contextmanager
def set_fftw_threads(n):
    try:
        import pyfftw
    except ImportError:
        yield
        return

    pyfftw_threads = pyfftw.config.NUM_THREADS
    try:
        pyfftw.config.NUM_THREADS = n
        yield
    finally:
        pyfftw.config.NUM_THREADS = pyfftw_threads


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


@contextmanager
def set_num_threads(n):
    import threadpoolctl
    with threadpoolctl.threadpool_limits(n), set_fftw_threads(n),\
            set_torch_threads(n), set_numba_threads(n):
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
