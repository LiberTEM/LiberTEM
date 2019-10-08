.. _`dataset api`:

Data Set API
------------

This API allows to load and handle data on a distributed system efficiently.

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
