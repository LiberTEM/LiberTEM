.. _`loading data`:

Loading Data
============

To efficiently handle files larger than main memory, LiberTEM never loads the whole
data set into memory. Calling the :meth:`~libertem.api.Context.load` function only opens the data set and gives
back a handle; running a job with :meth:`~libertem.api.Context.run` or :meth:`~libertem.api.Context.run_udf` then streams the data from mass storage.

There are two main ways of opening a data set in LiberTEM: using the GUI, or
the Python API.

Loading through the API
~~~~~~~~~~~~~~~~~~~~~~~

In the API, you can use :meth:`libertem.api.Context.load`. The general
pattern is:

.. code-block:: python

   >> ctx = Context()
   >> ctx.load("typename", path="/path/to/some/file", arg1="val1", arg2=42)

So, you need to specify the data set type, the path, and dataset-specific arguments. These
arguments are documented below.

For the full list of supported file formats and type names, see the documentation of :meth:`libertem.api.Context.load`.

.. _`Loading using the GUI`:

Loading using the GUI
~~~~~~~~~~~~~~~~~~~~~

Using the GUI, mostly the same parameters need to be specified, although some are only available
in the Python API. Tuples (for example for `scan_size` or `tileshape`) have to be entered as
comma-separated values. We follow the numpy convention here and specify the "fast-access"
dimension last, so a value of "42, 21" would mean the same as specifying (42, 21) in the
Python API, setting y=42 and x=21. Note that the GUI currently only support 4D data sets,
while the scripting API should handle more general n-dimensional data.

See also :doc:`the concepts section <concepts>`.

Common parameters
~~~~~~~~~~~~~~~~~

There are some common parameters across data set types:

 * name: the name of the data set, for display purposes
 * tileshape: some data set types support setting a tile shape, which is
   a tuning parameter. It can mostly be ignored and left to the default value,
   but sometimes you'll need to set it manually (for example for raw data
   sets). The tile shape is the smallest unit of data we are reading and
   working on.
 * scan_size: we generally support data containing rectangular 2D scans. For
   some data set types, you can specify a scan_size as a tuple (y, x); for
   others it is even required.

Supported Formats
~~~~~~~~~~~~~~~~~

LiberTEM supports the following file formats out of the box:

Merlin Medipix (MIB)
--------------------

.. autoclass:: libertem.io.dataset.mib.MIBDataSet

Raw binary files
----------------

.. autoclass:: libertem.io.dataset.raw.RawFileDataSet

EMPAD
-----

.. autoclass:: libertem.io.dataset.empad.EMPADDataSet

K2IS
----

.. autoclass:: libertem.io.dataset.k2is.K2ISDataSet

FRMS6
-----

.. autoclass:: libertem.io.dataset.frms6.FRMS6DataSet

BLO
---

.. autoclass:: libertem.io.dataset.blo.BloDataSet

SER
---

.. autoclass:: libertem.io.dataset.ser.SERDataSet

HDF5
----

.. autoclass:: libertem.io.dataset.hdf5.H5DataSet
