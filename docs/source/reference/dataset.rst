.. _`dataset api`:

Data Set API
============

This API allows to load and handle data on a distributed system efficiently. Note that you should
not directly use most dataset methods, but rather use the more high-level tools available, for
example user-defined functions.

See :ref:`our documentation on loading data <loading data>` for a high-level introduction.

.. _`formats`:

Formats
-------

.. _`mib`:

Merlin Medipix (MIB)
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: libertem.io.dataset.mib.MIBDataSet

.. _`raw binary`:

Raw binary files
~~~~~~~~~~~~~~~~

.. autoclass:: libertem.io.dataset.raw.RawFileDataSet

.. _`raw csr`:

Raw binary files in sparse CSR format
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: libertem.io.dataset.raw_csr.RawCSRDataSet

.. _`npy format`:

NumPy files (NPY)
~~~~~~~~~~~~~~~~~

.. autoclass:: libertem.io.dataset.npy.NPYDataSet

.. _`dm format`:

Digital Micrograph (DM3, DM4) files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There are currently two Digital Micrograph dataset implementations,
:class:`~libertem.io.dataset.dm_single.SingleDMDataSet` for a single-file,
C-ordered `.dm4` nD-dataset, and :class:`~libertem.io.dataset.dm.StackedDMDataSet`
for a stack of individual `.dm3` or `.dm4` image files which altogether
comprise a nD-dataset.

Both forms can be created using the following call to the :code:`Context`:

.. code-block::

   ctx.load('dm', ...)


and where possible the choice of reader (single-file or stacked) will
be inferred from the parameters.

.. autoclass:: libertem.io.dataset.dm_single.SingleDMDataSet

.. autoclass:: libertem.io.dataset.dm.StackedDMDataSet

DM4 datsets stored in a transposed format :code:`(sig, nav)` can
be converted to C-ordered data compatible with LiberTEM using the contrib function
:meth:`~libertem.contrib.convert_transposed.convert_dm4_transposed`.

.. _`empad`:

EMPAD
~~~~~

.. autoclass:: libertem.io.dataset.empad.EMPADDataSet

.. _`k2is`:

K2IS
~~~~

.. autoclass:: libertem.io.dataset.k2is.K2ISDataSet

.. _`frms6`:

FRMS6
~~~~~

.. autoclass:: libertem.io.dataset.frms6.FRMS6DataSet

.. _`blo`:

BLO
~~~

.. autoclass:: libertem.io.dataset.blo.BloDataSet

.. _`ser`:

SER
~~~

.. autoclass:: libertem.io.dataset.ser.SERDataSet

.. _`hdf5`:

HDF5
~~~~

.. autoclass:: libertem.io.dataset.hdf5.H5DataSet

.. _`seq`:

Norpix SEQ
~~~~~~~~~~

.. autoclass:: libertem.io.dataset.seq.SEQDataSet

.. _`mrc`:

MRC
~~~

.. autoclass:: libertem.io.dataset.mrc.MRCDataSet

.. _`tvips`:

TVIPS
~~~~~

.. autoclass:: libertem.io.dataset.tvips.TVIPSDataSet

.. _`memory`:

Memory data set
~~~~~~~~~~~~~~~

.. autoclass:: libertem.io.dataset.memory.MemoryDataSet

.. _`daskds`:

Dask
~~~~

.. autoclass:: libertem.io.dataset.dask.DaskDataSet

Converters
----------

.. autofunction:: libertem.contrib.convert_transposed.convert_dm4_transposed

Internal DataSet API
--------------------

.. automodule:: libertem.io.dataset.base
   :members:
   :undoc-members:
