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
delay the processing. This is highly experimental.

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

Dask integration
................

.. versionadded:: 0.9.0

By default, LiberTEM keeps the default Dask scheduler as-is and only
uses the Dask :code:`Client` internally to make sure existing workflows keep running
as before. For a closer integration it can be beneficial to use the same scheduler
for both LiberTEM and other Dask computations. There are several options for that:

:Set LiberTEM Dask cluster as default scheduler:
    * Use :code:`Context.make_with('dask-make-default')`
    * Pass :code:`client_kwargs={'set_as_default': True}` to
      :meth:`~libertem.executor.dask.DaskJobExecutor.connect` or
      :meth:`~libertem.executor.dask.DaskJobExecutor.make_local`
:Use existing Dask scheduler:
    * Use :code:`Context.make_with('dask-integration')` to start an executor
      that is compatible with the current Dask scheduler.
:Use dask.delayed:
    * Highly experimental! :class:`libertem.executor.delayed.DelayedJobExecutor` can
      return UDF computations as dask.delayed objects.

Reference
---------

For a full reference, please see :ref:`reference`.

.. note::
    The features described below are experimental and under development.

.. _daskarray:

Load datasets as Dask arrays
----------------------------

The :meth:`~libertem.contrib.daskadapter.make_dask_array` function can generate a `distributed Dask array <https://docs.dask.org/en/latest/array.html>`_ from a :class:`~libertem.io.dataset.base.DataSet` using its partitions as blocks. The typical LiberTEM partition size is close to the optimum size for Dask array blocks under most circumstances. The dask array is accompanied with a map of optimal workers. This map should be passed to the :meth:`compute` method in order to construct the blocks on the workers that have them in local storage.

.. NOTE: keep in sync with tests/io/test_dask_array.py::test_dask_array_2
.. code-block:: python

    from libertem.contrib.daskadapter import make_dask_array

    # Construct a Dask array from the dataset
    # The second return value contains information
    # on workers that hold parts of a dataset in local
    # storage to ensure optimal data locality
    dask_array, workers = make_dask_array(dataset)

    # Use the Dask.distributed client of LiberTEM, since it may not be
    # the default client:
    result = ctx.executor.client.compute(
        dask_array.sum(axis=(-1, -2))
    ).result()

In addition, Dask arrays can be interpreted as LiberTEM datasets under certain conditions
through use of the :meth:`~libertem.io.datasets.dask.DaskDataSet` wrapper class. This is
only likely to lead to good performance when the Dask array chunks are created through
lazy I/O or functions, via dask.delayed or similar routes. See :ref:`daskds` for details.


Run UDFs
--------

.. note::
    The features described here are experimental and under development.
Compute UDFs with dask.delayed
------------------------------

.. versionadded:: 0.9.0

Using a :class:`~libertem.executor.delayed.DelayedJobExecutor` with a
:class:`~libertem.api.Context` lets :class:`~libertem.api.Context.run_udf`
return a dask.delayed value for the result. The computation is only
performed when the :code:`compute()` method is called on it. Please note
that the dask.delayed values generated this way are not Dask arrays, but
delayed NumPy arrays. In particular, they are not chunked.

Generate chunked Dask arrays with UDFs
--------------------------------------

.. versionadded:: 0.9.0

The :meth:`~libertem.contrib.daskadapter.task_results_array` function can
generate chunked Dask arrays from intermediate UDF task results, i.e.
before merging and results computation.

Computing the final result of an UDF from this intermediate Dask array requires
implementing a custom, equivalent merge and results computation routine since
LiberTEM relies heavily on modification of buffer slices for this step. This is
incompatible with Dask arrays.

For most UDFs, an equivalent implementation that is compatible with
Dask arrays should be easy to implement. Inspect an UDF's implementation of
:code:`merge()` and/or :code:`get_results()` to see how the final result is calculated.
For the default merging of :code:`kind="nav"` buffers, the equivalent method is just
reshaping the resulting array into the correct shape.

Simple merge example
....................

.. testsetup:: daskarray

    import numpy as np
    from pprint import pprint

    from libertem import api as lt

    ctx = lt.Context.make_with('inline')

    dataset = ctx.load(
        'memory',
        data=np.random.random((13, 14, 15, 16)),
        num_partitions=4,
    )

:class:`~libertem.udf.masks.ApplyMasksUDF` returns a single :code:`kind="nav"`
buffer with default merge. The result from
:meth:`~libertem.contrib.daskadapter.task_results_array` is very similar to the
UDF result, consequently.

.. testcode:: daskarray

    from libertem.udf.masks import ApplyMasksUDF
    from libertem.contrib.daskadapter import task_results_array

    udf = ApplyMasksUDF(
        mask_factories=[lambda: np.ones(dataset.shape.sig)]
    )

    ref = ctx.run_udf(dataset=dataset, udf=udf)
    pprint(ref)

.. testoutput:: daskarray

    {'intensity': <BufferWrapper kind=nav dtype=float64 extra_shape=(1,)>}

.. testcode:: daskarray

    res = task_results_array(dataset=dataset, udf=udf)
    pprint(res)

.. testoutput:: daskarray

    {'intensity': dask.array<concatenate, shape=(182, 1), dtype=float64, chunksize=(45, 1), chunktype=numpy.ndarray>}

.. testcode:: daskarray

    # The merge function is trivial, just reshape
    target_shape = (*dataset.shape.nav, 1)

    print(np.allclose(
        ref['intensity'].data,
        res['intensity'].reshape(target_shape).compute()
    ))

.. testoutput:: daskarray

    True

Complex merge example
.....................

This example shows :class:`~ libertem.udf.stddev.StdDevUDF` which has a complex
merge function. The results from running the UDF and
:meth:`~libertem.contrib.daskadapter.task_results_array` are quite different.
Implementing an equivalent merge would require a close look at
the implementation of :class:`~ libertem.udf.stddev.StdDevUDF`.

.. testcode:: daskarray

    from libertem.udf.stddev import StdDevUDF
    from libertem.contrib.daskadapter import task_results_array

    ref = ctx.run_udf(dataset=dataset, udf=StdDevUDF())
    pprint(ref)

.. testoutput:: daskarray

    {'mean': <BufferWrapper kind=sig dtype=float64 extra_shape=()>,
     'num_frames': <BufferWrapper kind=single dtype=int32 extra_shape=()>,
     'std': <BufferWrapper kind=sig dtype=float64 extra_shape=()>,
     'sum': <BufferWrapper kind=sig dtype=float64 extra_shape=()>,
     'var': <BufferWrapper kind=sig dtype=float64 extra_shape=()>,
     'varsum': <BufferWrapper kind=sig dtype=float64 extra_shape=()>}

.. testcode:: daskarray

    res = task_results_array(dataset=dataset, udf=StdDevUDF())
    # Note that several result buffers are missing here since they are
    # only populated in get_results().
    pprint(res)

.. testoutput:: daskarray

    {'num_frames': dask.array<stack, shape=(5, 1), dtype=int32, chunksize=(1, 1), chunktype=numpy.ndarray>,
     'sum': dask.array<stack, shape=(5, 15, 16), dtype=float64, chunksize=(1, 15, 16), chunktype=numpy.ndarray>,
     'varsum': dask.array<stack, shape=(5, 15, 16), dtype=float64, chunksize=(1, 15, 16), chunktype=numpy.ndarray>} 

.. testcode:: daskarray

    # The merge function for the sum is simple
    print(np.allclose(ref['sum'].raw_data, res['sum'].sum(axis=0).compute()))

.. testoutput:: daskarray

    True

.. testcode:: daskarray

    # However, the merge function for the sum of the variance is not trivial.
    # It would require a deep look into StdDevUDF to re-implement it.
    print(np.allclose(ref['varsum'].raw_data, res['varsum'].sum(axis=0).compute()))

.. testoutput:: daskarray

    False
