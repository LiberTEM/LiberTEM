.. _`api documentation`:

Python API
==========

The Python API is a concise API for using LiberTEM from Python code. It is suitable both
for interactive scripting, for example from Jupyter notebooks, and for usage
from within a Python application or script.

Basic example
-------------

This is a basic example to load the API, create a local cluster, load a file and run a job. For a complete example on how to use the Python API, please see the
Jupyter notebooks in `the example directory <https://github.com/LiberTEM/LiberTEM/tree/master/examples>`_.

For more details on loading data and a reference of supported file formats, please see :ref:`loading data`.

.. include:: /../../examples/basic.py
    :code:

For a full API reference, please see :ref:`reference`.

To go beyond the included capabilities of LiberTEM, you can implement your own using :ref:`user-defined functions`.

.. _daskarray:

Integration with Dask arrays
----------------------------

The :meth:`~libertem.contrib.dask.make_dask_array` function can generate a `distributed Dask array <https://docs.dask.org/en/latest/array.html>`_ from a :class:`~libertem.io.dataset.base.DataSet` using its partitions as blocks. The typical LiberTEM partition size is close to the optimum size for Dask array blocks under most circumstances. The dask array is accompanied with a map of optimal workers. This map should be passed to the :meth:`compute` method in order to construct the blocks on the workers that have them in local storage.

.. include:: /../../examples/dask_array.py
    :code:
