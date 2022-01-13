.. _`api documentation`:

Python API
==========

The Python API is a concise API for using LiberTEM from Python code. It is suitable both
for interactive scripting, for example from Jupyter notebooks, and for usage
from within a Python application or script.

Basic example
-------------

This is a basic example to load the API, create a local cluster, load a file and
run an analysis. For complete examples on how to use the Python API, please see the
Jupyter notebooks in `the example directory
<https://github.com/LiberTEM/LiberTEM/tree/master/examples>`_.

For more details, please see :ref:`loading data`, :ref:`dataset api` and
:ref:`format-specific reference<formats>`. See :ref:`sample data` for publicly available datasets.

.. include:: /../../examples/basic.py
    :code:

Custom processing routines
--------------------------

To go beyond the included capabilities of LiberTEM, you can implement your own
using :ref:`user-defined functions`.

Reference
---------

For a full reference, please see :ref:`reference`.

.. _`executors`:

Executors
---------

.. versionadded:: 0.9.0
    The executor API is internal. Since choice and parameters of executors
    are important for integration with Dask and other frameworks,
    they are now documented. Only the names and creation methods for
    executors are reasonably stable. The rest of the API is subject to
    change without notice. For that reason it is documented in the developer
    section and not in the API reference.

The default executor is :class:`~libertem.executor.dask.DaskJobExecutor` that
uses the dask.distributed scheduler. To support all LiberTEM features and
achieve optimal performance, the methods provided by LiberTEM to start a
dask.distributed cluster should be used. However, LiberTEM can also run on a
"vanilla" dask.distributed cluster.

The :class:`~libertem.executor.inline.InlineJobExecutor` runs all tasks
synchronously in the current thread. This is useful for debugging and for
special applications such as running UDFs that perform their own multithreading
efficiently or for other non-standard use that requires tasks to be executed
sequentially and in order.

.. versionadded:: 0.9.0

The :class:`~libertem.executor.concurrent.ConcurrentJobExecutor` runs all tasks
using :mod:`python.concurrent.futures`. Using the
:class:`python:concurrent.futures.ThreadPoolExecutor` allows
sharing large amounts of data as well as other resources between main thread
and workers efficiently, but is severely slowed down by the Python
`global interpreter lock <https://wiki.python.org/moin/GlobalInterpreterLock>`_
under many circumstances.

For special applications, the :class:`~libertem.executor.delayed.DelayedJobExecutor`
can use `dask.delayed <https://docs.dask.org/en/stable/delayed.html>`_ to
delay the processing. This is experimental, see :ref:`dask` for more details.

Common executor choices
.......................

.. versionadded:: 0.9.0

:meth:`libertem.api.Context.make_with` provides a convenient shortcut to start a
:class:`~libertem.api.Context` with common executor choices. See the
:meth:`API documentation <libertem.api.Context.make_with>`
for available options!

Connect to a cluster
....................

See :ref:`cluster` on how to start a scheduler and workers.

.. Not run with docs-check since it doesn't play well with launching a multiprocessing
   cluster

.. code-block:: python

    from libertem import api
    from libertem.executor.dask import DaskJobExecutor

    # Connect to a Dask.Distributed scheduler at 'tcp://localhost:8786'
    with DaskJobExecutor.connect('tcp://localhost:8786') as executor:
        ctx = api.Context(executor=executor)
        ...

.. _`cluster spec`:

Customize CPUs and CUDA devices
...............................

To control how many CPUs and which CUDA devices are used, you can specify them as follows:

.. Not run with docs-check since it doesn't play well with launching a multiprocessing
   cluster

.. code-block:: python

    from libertem import api
    from libertem.executor.dask import DaskJobExecutor, cluster_spec
    from libertem.utils.devices import detect

    # Find out what would be used, if you like
    # returns dictionary with keys "cpus" and "cudas", each with a list of device ids
    devices = detect()

    # Example: Deactivate CUDA devices by removing them from the device dictionary
    devices['cudas'] = []

    # Example: Deactivate CuPy integration
    devices['has_cupy'] = False

    # Example: Use 3 CPUs. The IDs are ignored at the moment, i.e. no CPU pinning
    devices['cpus'] = range(3)

    # Generate a spec for a Dask.distributed SpecCluster
    # Relevant kwargs match the dictionary entries
    spec = cluster_spec(**devices)
    # Start a local cluster with the custom spec
    with DaskJobExecutor.make_local(spec=spec) as executor:
        ctx = api.Context(executor=executor)
        ...

Please see :ref:`dask executor` for a reference of the Dask-based executor.
