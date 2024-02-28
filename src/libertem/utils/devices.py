import functools
import warnings
import logging
from typing_extensions import TypedDict

import psutil

import numba.cuda


logger = logging.getLogger(__name__)


try:
    import cupy
except ModuleNotFoundError:
    cupy = None
except ImportError as e:
    # Cupy can be a bit fragile; allow running LiberTEM with
    # messed-up installation
    warnings.warn(repr(e), RuntimeWarning)
    cupy = None


class DetectResult(TypedDict):
    cpus: list[int]
    cudas: list[int]
    has_cupy: bool


def detect() -> DetectResult:
    '''
    Detect which devices are present

    .. versionadded:: 0.6.0

    Returns
    -------
    dict
        Dictionary with keys :code:`'cpus'` and :code:`'cudas'`
        Each containing
        a list of devices. Only physical CPU cores are counted, i.e. no
        hyperthreading.
        Additionally it has the key :code:`'has_cupy'`, which signals
        if cupy is installed and available.
    '''
    cores = psutil.cpu_count(logical=False)
    if cores is None:
        cores = 2
    try:
        cudas = [device.id for device in numba.cuda.gpus]
    except numba.cuda.CudaSupportError as e:
        # Continue running without GPU or in case of errors
        cudas = []
        logger.info(repr(e))
    return {
        "cpus": list(range(cores)),
        "cudas": cudas,
        "has_cupy": has_cupy(),
    }


@functools.cache
def has_cupy():
    '''
    Probe if :code:`cupy` was loaded successfully.

    .. versionadded:: 0.6.0

    CuPy is an optional dependency with special integration for UDFs. See
    :ref:`udf cuda` for details.
    '''
    if cupy is None:
        return False
    try:
        cupy.cuda
        cupy.array(cupy.zeros((1,)))
        return True
    except Exception as e:  # possibly: AttributeError or CompileException
        warnings.warn(repr(e), RuntimeWarning)
        return False
