from typing import Union
from collections.abc import Iterable
import itertools
import warnings


def assign_cudas(cudas: Union[int, Iterable[int]]) -> Iterable[int]:
    """
    Takes the cudas argument to :code:`cluster_spec` and
    converts it into a properly formatted iterable of CUDA
    device ids

    If cudas is an integer, assigns this many workers to
    device ids in a round-robin fashion, where CUDA devices
    can be detected. If the devices cannot be detected raise
    a warning and assign device_ids in a sequential fashion.

    Will also raise a warning if cudas is a non-empty iterable
    on a system where the CUDA devices cannot be detected.
    """
    if isinstance(cudas, int) or len(cudas):
        # Needed to know if we can assign CUDA workers
        from libertem.utils.devices import detect
        avail_cudas = detect()['cudas']
        if not avail_cudas and cudas:  # needed in case cudas == 0
            warnings.warn('Specifying CUDA workers on system with '
                          'no visible CUDA devices',
                          RuntimeWarning)
            # If we are assigning from int, just use increasing
            # device indices even if they are unavailable
            avail_cudas = itertools.count()

        if isinstance(cudas, int):
            # Round-Robin-assign to available CUDA devices
            # Can override by specifying cudas as an iterable
            cudas_iter = itertools.cycle(avail_cudas)
            cudas = tuple(next(cudas_iter) for _ in range(cudas))

    return cudas
