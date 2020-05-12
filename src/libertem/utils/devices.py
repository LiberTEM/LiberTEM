import warnings

import psutil

try:
    import cupy
    import numba.cuda
except ModuleNotFoundError:
    cupy = None
except ImportError as e:
    # Cupy can be a bit fragile; allow running LiberTEM with
    # messed-up installation
    warnings.warn(repr(e), RuntimeWarning)
    cupy = None


def detect():
    cores = psutil.cpu_count(logical=False)
    if cores is None:
        cores = 2

    if cupy:
        try:
            cudas = [device.id for device in numba.cuda.gpus]
        except numba.cuda.CudaSupportError as e:
            # Continue running without GPU in case of errors
            # Keep LiberTEM usable with misconfigured CUDA, CuPy or numba.cuda
            # This DOES happen, ask @uellue!
            cudas = []
            warnings.warn(repr(e), RuntimeWarning)
    else:
        cudas = []
    return {
        "cpus": range(cores),
        "cudas": cudas
    }
