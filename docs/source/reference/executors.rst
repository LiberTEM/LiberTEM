.. _`executor api`:

LiberTEM executors
==================

See :ref:`executors` for a general overview.

.. versionadded:: 0.9.0
    Previously, the executor API was internal. Since influence on the executor
    is important for integration with Dask and other frameworks,
    the API is now documented to help with that. Nevertheless, this is still
    experimental and may change between releases without notice.

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
