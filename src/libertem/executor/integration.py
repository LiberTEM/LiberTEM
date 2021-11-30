import types

import dask
import dask.delayed
import dask.distributed as dd

from .dask import DaskJobExecutor
from .concurrent import ConcurrentJobExecutor
from .inline import InlineJobExecutor


def get_dask_integration_executor():
    '''
    Query the current Dask scheduler and return a :class:`~libertem.executor.base.JobExecutor`
    that is compatible with it.

    .. versionadded:: 0.9.0

    If a dask.distributed :code:`Client` is set as a scheduler, use it with a
    :class:`~libertem.executor.dask.DaskJobExecutor`.
    '''
    item = dask.delayed(1)
    dask_scheduler = dask.base.get_scheduler(collections=[item])
    # We first test for circumstances where we know how to return an adapted
    # JobExecutor instance.
    if isinstance(dask_scheduler, types.MethodType):
        if isinstance(dask_scheduler.__self__, dd.Client):
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
