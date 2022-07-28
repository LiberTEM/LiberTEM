.. _`executor api`:

LiberTEM executors
==================

All access to data and processing is done by an executor that implements the
:class:`~libertem.common.executor.JobExecutor` interface to run functions and
tasks. The executor specifies where and how processing is done, including
running on a cluster or in a single thread, while being independent of other
parts of LiberTEM. See :ref:`executors` for an overview from a user's perspective.

.. versionadded:: 0.9.0
    The executor API is internal. Since choice and parameters of executors
    are important for integration with Dask and other frameworks,
    they are now documented. Only the names and creation methods for
    executors are reasonably stable. The rest of the API is subject to
    change without notice.

Base classes
................

.. automodule:: libertem.common.executor
    :members: JobExecutor, AsyncJobExecutor

.. _`dask executor`:

Dask.Distributed
................

The :class:`~libertem.executor.dask.DaskJobExecutor` is the default executor
when creating a :class:`~libertem.api.Context` with no parameters.

.. automodule:: libertem.executor.dask
    :members:

.. automodule:: libertem.executor.integration
    :members:

Inline
......

.. automodule:: libertem.executor.inline
    :members:

Concurrent
..........

.. automodule:: libertem.executor.concurrent
    :members:

Dask.delayed
............

See :ref:`delayed_udfs` for more information about 
:class:`~libertem.executor.delayed.DelayedJobExecutor`.

.. automodule:: libertem.executor.delayed
    :members:

.. _`pipelined executor`:

Pipelined executor
..................

.. automodule:: libertem.executor.pipelined
    :members:
