.. _`dataset api`:

Data Set API
------------

This API allows to load and handle data on a distributed system efficiently. Note that you should
not directly use most dataset methods, but rather use the more high-level tools available, for
example user-defined functions.

See :ref:`our documentation on loading data <loading data>` for a high-level introduction.

.. automodule:: libertem.io.dataset.base
   :members:
   :undoc-members:
   :special-members: __init__

.. _`formats`:
.. _`mib`:

Merlin Medipix (MIB)
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: libertem.io.dataset.mib.MIBDataSet

.. _`raw binary`:

Raw binary files
~~~~~~~~~~~~~~~~

.. autoclass:: libertem.io.dataset.raw.RawFileDataSet

.. _`dm format`:

Digital Micrograph (DM3, DM4) files
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: libertem.io.dataset.dm.DMDataSet

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

.. _`memory`:

Memory data set
~~~~~~~~~~~~~~~

.. autoclass:: libertem.io.dataset.memory.MemoryDataSet
    :special-members: __init__
