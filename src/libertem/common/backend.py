import os
import platform
from typing import Optional


'''
Get and set environment variables that indicate to use CPU or GPU processing and
the appropriate device IDs

This allows configuring a Dask worker or inline executor to use CPU or a
particular GPU, which allows simultaneous use of all GPUs and CPUs on a
processing node without oversubscription.

Internally, this gets and sets the environment variables
:code:`LIBERTEM_USE_CUDA` and :code:`LIBERTEM_USE_CPU`.

See :ref:`udf cuda` for details on how to run UDFs with CUDA support and
:ref:`debugging udfs` for information on how to use these functions to control
where an :class:`~libertem.executor.inline.InlineJobExecutor` runs tasks.
'''


def get_use_cuda() -> Optional[int]:
    '''
    .. versionadded:: 0.6.0

    Returns
    -------
    id : int or None
        CUDA device ID to use.
    '''
    ret = os.environ.get("LIBERTEM_USE_CUDA")
    if ret is not None:
        return int(ret)
    else:
        return None


def set_use_cuda(cuda_device: int):
    '''
    This sets a CUDA device ID and unsets any CPU ID

    .. versionadded:: 0.6.0

    Parameters
    ----------
    cuda_device : int
        CUDA device ID to use
    '''
    os.environ["LIBERTEM_USE_CUDA"] = str(cuda_device)
    os.environ.pop("LIBERTEM_USE_CPU", None)


def get_use_cpu():
    '''
    .. versionadded:: 0.6.0

    Returns
    -------
    id : int or None
        CPU device ID to use. Currently there is no pinning, i.e. the value itself is
        ignored. :code:`None` means "don't use CPU" and any integer means "use CPU".
        Default is 0 if no settings were applied before.
    '''
    ret = os.environ.get("LIBERTEM_USE_CPU")
    if ret is not None:
        ret = int(ret)
    elif get_use_cuda() is None:
        # If no variable is set, return CPU 0
        # For example inline executor or test code
        ret = 0
    return ret


def set_use_cpu(cpu: int):
    '''
    This sets a CPU device ID and unsets any CUDA ID

    .. versionadded:: 0.6.0

    Parameters
    ----------
    cpu : int
        CPU to use. The value is currently ignored, i.e. any CPU is used without pinning
    '''
    os.environ.pop("LIBERTEM_USE_CUDA", None)
    os.environ["LIBERTEM_USE_CPU"] = str(cpu)


def get_device_class():
    '''
    .. versionadded:: 0.6.0

    Returns
    -------
    class : str
        Device class to use. Can be 'cpu' or 'cuda'. Default is 'cpu' if
        no settings were applied before.
    '''
    cuda = get_use_cuda()
    cpu = get_use_cpu()
    if cpu is not None and cuda is not None:
        raise RuntimeError(
            "Both LIBERTEM_USE_CPU and LIBERTEM_USE_CUDA set, expecting at most one"
        )
    if cuda is not None:
        return "cuda"
    else:
        return "cpu"


def set_file_limit():
    if platform.system() == "Windows":
        # Windows has no problem opening huge numbers of files
        return

    import resource
    _, hard_lim = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (hard_lim, hard_lim))
