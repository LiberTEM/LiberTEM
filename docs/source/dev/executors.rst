.. _`executor api`:

LiberTEM executors
==================

All access to data and processing is done by an executor that implements the
:class:`~libertem.executor.base.JobExecutor` interface to run functions and
tasks. That allows to modify where and how processing is done, including running
on a cluster or in a single thread, without changes in other parts of LiberTEM.
See :ref:`executors` for an overview from a user's perspective.

.. versionadded:: 0.9.0
    The executor API is internal. Since choice and parameters of executors
    are important for integration with Dask and other frameworks,
    they are now documented. Only the names and creation methods for
    executors are reasonably stable. The rest of the API is subject to
    change without notice.

.. _`dask executor`:

Dask.Distributed
................

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

.. automodule:: libertem.executor.delayed
    :members:
