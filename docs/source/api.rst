.. _`api documentation`:

Python API
==========

The Python API is a concise API for using LiberTEM from Python code. It is suitable both
for interactive scripting, for example from Jupyter notebooks, and for usage
from within a Python application or script.

For a full API reference, please see :ref:`reference`.

.. _`context`:

Context
-------

The :class:`libertem.api.Context` object is the entry-point for most interaction
and processing with LiberTEM. It is used to load datasets, specify and run analyses.
The following snippet initializes a :class:`~libertem.api.Context` ready-for-use
with default parameters and backed by a parallel processing engine.

.. code-block:: python

    import libertem.api as lt

    with lt.Context() as ctx:
        ...

See the :class:`API documentation <libertem.api.Context>` for the full capabilities
exposed by the :class:`~libertem.api.Context`.

.. note::
    The use of a :code:`with` block in the above code ensures the :class:`~libertem.api.Context`
    will correctly release any resources it holds on to when it goes out of scope,
    but it is also possible to use the object as a normal variable, i.e. :code:`ctx = lt.Context()`

Basic example
-------------

This is a basic example to load the API, create a local cluster, load a file and
run an analysis.

.. include:: /../../examples/basic.py
    :code:

For complete examples on how to use the Python API, please see the
Jupyter notebooks in `the example directory
<https://github.com/LiberTEM/LiberTEM/tree/master/examples>`_.

For more details on the data formats that LiberTEM supports, please see
:ref:`loading data`, :ref:`dataset api` and :ref:`format-specific reference<formats>`.
See :ref:`sample data` for publicly available datasets.

Custom processing routines
--------------------------

To go beyond the included capabilities of LiberTEM, you can implement your own
analyses using :ref:`user-defined functions`. UDFs are dataset-agnostic
and benefit from the same parallelisation as the built-ins tools.

.. _`executors`:

Executors
---------

An Executor is the internal engine which the :class:`~libertem.api.Context` uses to
compute user-defined functions or run other tasks. Executors can be serial or parallel,
and can differ substantially in their implementation, but all adhere to a
common interface which the :class:`~libertem.api.Context` understands.

.. versionadded:: 0.9.0
    The executor API is internal. Since choice and parameters of executors
    are important for integration with Dask and other frameworks,
    they are now documented. Only the names and creation methods for
    executors are reasonably stable. The rest of the API is subject to
    change without notice. For that reason it is documented in the developer
    section and not in the API reference.

The default executor is :class:`~libertem.executor.dask.DaskJobExecutor` that
uses the :code:`dask.distributed` scheduler. To support all LiberTEM features and
achieve optimal performance, the methods provided by LiberTEM to start a
:code:`dask.distributed` cluster should be used. However, LiberTEM can also run on a
"vanilla" :code:`dask.distributed` cluster. Please note that :code:`dask.distributed`
clusters that are not created by LiberTEM might use threading or a mixture of threads
and processes, and therefore might behave or perform differently to a
LiberTEM-instantiated cluster.

The :class:`~libertem.executor.inline.InlineJobExecutor` runs all tasks
synchronously in the current thread. This is useful for debugging and for
special applications such as running UDFs that perform their own multithreading
efficiently or for other non-standard use that requires tasks to be executed
sequentially and in order.

See also :ref:`threading` for more information on multithreading in UDFs.

.. versionadded:: 0.9.0

The :class:`~libertem.executor.concurrent.ConcurrentJobExecutor` runs all tasks
using :mod:`python:concurrent.futures`. Using a
:class:`python:concurrent.futures.ThreadPoolExecutor`, which is the deafult behaviour,
allows sharing large amounts of data as well as other resources between the
main thread and workers efficiently, but is severely slowed down by the
Python `global interpreter lock <https://wiki.python.org/moin/GlobalInterpreterLock>`_
under many circumstances. Furthermore, it can create thread safety issues such as
https://github.com/LiberTEM/LiberTEM-blobfinder/issues/35.

It is also in principle possible to use a :class:`python:concurrent.futures.ProcessPoolExecutor`
as backing for the :class:`~libertem.executor.concurrent.ConcurrentJobExecutor`,
though this is untested and is likely to lead to worse performance than the
LiberTEM default :class:`~libertem.executor.dask.DaskJobExecutor`.

For special applications, the
:class:`~libertem.executor.delayed.DelayedJobExecutor` can use `dask.delayed
<https://docs.dask.org/en/stable/delayed.html>`_ to delay the processing. This
is experimental, see :ref:`dask` for more details. It might use threading as
well, depending on the Dask scheduler that is used by :code:`compute()`.

.. _`pipelined`:

Pipelined executor
------------------

.. versionadded:: 0.10.0

For live data processing using
`LiberTEM-live <https://libertem.github.io/LiberTEM-live/>`_, the
:class:`~libertem.executor.pipelined.PipelinedExecutor`
provides a multiprocessing executor that routes the live data source in a round-robin
fashion to worker processes. This is important to support processing that cannot keep
up with the detector speed on a single CPU core. This executor also works for offline
data sets in principle, but is not optimized for that use case.

Please see :ref:`pipelined executor` for a reference of the pipelined executor,
and `the LiberTEM-live documentation <https://libertem.github.io/LiberTEM-live/>`_
for details on live processing.


.. _`cluster spec`:

Specifying executor type, CPU and GPU workers
.............................................

.. versionadded:: 0.9.0

:meth:`libertem.api.Context.make_with` provides a convenient shortcut to start a
:class:`~libertem.api.Context` with specific executor and customise the number of
workers it uses.

.. code-block:: python

    import libertem.api as lt

    # Create a Dask-based Context with 4 cpu workers and 2 gpu workers
    with lt.Context.make_with('dask', cpus=4, gpus=2) as ctx:
        ...

The default behaviour is to create a Dask-based Context, but the same
method can be used to create any executor, as described in the documentation
of the method. A useful shortcut is :code:`lt.Context.make_with('inline')`
to quickly create a synchronous executor for debugging.

.. note::
    Not all executor types allow specifying number of workers, and
    not all executor types are GPU-capable. In these cases the :code:`make_with`
    method will raise an :class:`~libertem.exceptions.ExecutorSpecException`.

See the :meth:`API documentation <libertem.api.Context.make_with>`
for more information.


Connect to an existing cluster
..............................

The :class:`~libertem.executor.dask.DaskJobExecutor` is capable of connecting
to an existing :code:`dask.distributed` scheduler, which may be a centrally managed
installation on a physical cluster, or a local, single-machine scheduler started for
some other purpose (by LiberTEM or directly through Dask). Cluster re-use can
reduce startup times as there is no requirement to spawn new workers each time
a script or Notebook is executed.

See :ref:`cluster` for more on how to start a scheduler and workers.

.. Not run with docs-check since it doesn't play well with
   launching a multiprocessing cluster

.. code-block:: python

    import libertem.api as lt
    from libertem.executor.dask import DaskJobExecutor

    # Connect to a Dask.Distributed scheduler at 'tcp://localhost:8786'
    with DaskJobExecutor.connect('tcp://localhost:8786') as executor:
        ctx = lt.Context(executor=executor)
        ...
