import os
import logging
from pathlib import Path
from ctypes.util import find_library
import ctypes

log = logging.getLogger(__name__)


def set_num_threads(n):
    # The library has to be loaded for each call
    # through ctypes
    # and not once during import of this module
    # for this to work with dask.distributed workers

    def get_numpy_openblas_dll_path():
        import numpy
        numpy_dir = os.path.dirname(numpy.__file__)
        dll_dir = os.path.join(numpy_dir, '.libs')
        files = Path(dll_dir).rglob('libopenblas*.dll')
        return str(next(files))

    try:
        if os.name == 'nt':
            lib_str = get_numpy_openblas_dll_path()
        else:
            lib_str = find_library('openblas')
        openblas = ctypes.cdll.LoadLibrary(lib_str)

        openblas_set_num_threads = openblas.openblas_set_num_threads
        log.debug("Found OpenBLAS library")
    except Exception as e:
        log.warning("Didn't find OpenBLAS library", exc_info=e)

        def openblas_set_num_threads(n):
            pass

    openblas_set_num_threads(n)
