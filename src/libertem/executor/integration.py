import types
import warnings

import dask
import dask.delayed
import dask.distributed as dd

from .dask import DaskJobExecutor
from .concurrent import ConcurrentJobExecutor
from .inline import InlineJobExecutor


def get_dask_integration_executor():
    '''
    Query the current Dask scheduler and return a :class:`~libertem.common.executor.JobExecutor`
    that is compatible with it. See https://docs.dask.org/en/stable/scheduling.html
    for the meaning of the different scheduler types.

    This can be used to integrate LiberTEM in an existing Dask workflow. This
    may not achieve optimal LiberTEM performance and will usually not allow GPU
    processing with LiberTEM, but avoids potential compatibility issues from
    changing or duplicating the Dask scheduler in an existing workflow.

    .. versionadded:: 0.9.0

    If a :code:`dask.distributed.Client` is set as the scheduler, return a
    :class:`~libertem.executor.dask.DaskJobExecutor` using this :code:`Client`.

    If the Dask scheduler is :code:`'threads'`, return a
    :class:`~libertem.executor.concurrent.ConcurrentJobExecutor` backed
    by the same thread pool as used by Dask.

    If the Dask scheduler is :code:`'synchronous'`, return an
    :class:`~libertem.executor.inline.InlineJobExecutor`
    which mimics the single-process, single-thread behaviour of Dask.
    '''
    item = dask.delayed(1)
    dask_scheduler = dask.base.get_scheduler(collections=[item])
    # We first test for circumstances where we know how to return an adapted
    # JobExecutor instance.
    if isinstance(dask_scheduler, types.MethodType):
        if isinstance(dask_scheduler.__self__, dd.Client):
            # See https://github.com/dask/distributed/issues/6776
            if dask.config.get("distributed.worker.profile.enabled"):
                warnings.warn(
                    "Dask profiling seems to be enabled, which is known to cause issues with "
                    "the DM reader. "
                    "See https://github.com/dask/distributed/issues/6776"
                )
            return DaskJobExecutor(client=dask_scheduler.__self__)
    elif dask_scheduler is dask.threaded.get:
        if dask.threaded.default_pool:
            return ConcurrentJobExecutor(client=dask.threaded.default_pool)
    # ConcurrentJobExecutor is currently incompatible with ProcessPoolExecutor
    # since it can't pickle local functions.
    # Therefore, fall through to default case for now.
    # elif dask_scheduler is dask.multiprocessing.get:
    #     # FIXME more research needed if the pool is cached and if
    #     # one can get hold of it.
    #     return ConcurrentJobExecutor(
    #         client=concurrent.futures.ProcessPoolExecutor(),
    #         is_local=True
    #     )
    elif dask_scheduler is dask.local.get_sync:
        return InlineJobExecutor()

    # If we didn't return yet,
    # we fall through to the default case.
    return ConcurrentJobExecutor.make_local()
