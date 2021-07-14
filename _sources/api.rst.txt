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

Connect to a cluster
--------------------

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

Customize CPUs and CUDA devices
-------------------------------

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

For a full API reference, please see :ref:`reference`.

To go beyond the included capabilities of LiberTEM, you can implement your own using :ref:`user-defined functions`.

.. _daskarray:

Integration with Dask arrays
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
