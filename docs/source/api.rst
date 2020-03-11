.. _`api documentation`:

Python API
==========

The Python API is a concise API for using LiberTEM from Python code. It is suitable both
for interactive scripting, for example from Jupyter notebooks, and for usage
from within a Python application or script.

Basic example
-------------

This is a basic example to load the API, create a local cluster, load a file and
run an analysis. For a complete example on how to use the Python API, please see the
Jupyter notebooks in `the example directory
<https://github.com/LiberTEM/LiberTEM/tree/master/examples>`_.

For more details, please see :ref:`loading data`, :ref:`dataset api` and
:ref:`format-specific reference<formats>`.

.. include:: /../../examples/basic.py
    :code:

For a full API reference, please see :ref:`reference`.

To go beyond the included capabilities of LiberTEM, you can implement your own using :ref:`user-defined functions`.

.. _daskarray:

Integration with Dask arrays
----------------------------

The :meth:`~libertem.contrib.daskadapter.make_dask_array` function can generate a `distributed Dask array <https://docs.dask.org/en/latest/array.html>`_ from a :class:`~libertem.io.dataset.base.DataSet` using its partitions as blocks. The typical LiberTEM partition size is close to the optimum size for Dask array blocks under most circumstances. The dask array is accompanied with a map of optimal workers. This map should be passed to the :meth:`compute` method in order to construct the blocks on the workers that have them in local storage.

.. testsetup:: *

    from libertem import api
    from libertem.executor.inline import InlineJobExecutor

    ctx = api.Context(executor=InlineJobExecutor())
    dataset = ctx.load("memory", datashape=(16, 16, 16), sig_dims=2)

.. testcode::

    from libertem.contrib.daskadapter import make_dask_array

    # Construct a Dask array from the dataset
    # The second return value contains information
    # on workers that hold parts of a dataset in local
    # storage to ensure optimal data locality
    dask_array, workers = make_dask_array(dataset)

    # Perform calculations using the worker map.
    result = dask_array.sum(axis=(-1, -2)).compute(workers=workers)
