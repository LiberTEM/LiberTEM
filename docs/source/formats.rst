.. _`loading data`:

Loading data
============

To efficiently handle files larger than main memory, LiberTEM never loads the
whole data set into memory. Calling the :meth:`~libertem.api.Context.load`
function only opens the data set and gives back a handle; running an analysis
with :meth:`~libertem.api.Context.run` or a UDF with
:meth:`~libertem.api.Context.run_udf` then streams the data from mass storage.

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

For the full list of supported file formats and type names, see :ref:`format-specific API reference<formats>`.

.. _`Loading using the GUI`:

Loading using the GUI
~~~~~~~~~~~~~~~~~~~~~

Using the GUI, mostly the same parameters need to be specified, although some are only available
in the Python API. Tuples (for example for `scan_size` or `tileshape`) have to be entered as
comma-separated values. We follow the NumPy convention here and specify the "fast-access"
dimension last, so a value of `"42, 21"` would mean the same as specifying :code:`(42, 21)` in the
Python API, setting `y=42` and `x=21`. Note that the GUI currently only support 4D data sets,
while the scripting API should handle more general n-dimensional data.

See also :ref:`the concepts section <concepts>`.

Common parameters
~~~~~~~~~~~~~~~~~

There are some common parameters across data set types:

`name`
  The name of the data set, for display purposes. Only used in the GUI.
`tileshape`
  Some data set types support setting a tile shape, which is
  a tuning parameter. It can mostly be ignored and left to the default value,
  but sometimes you'll need to set it manually (for example for raw data
  sets). The tile shape is the smallest unit of data we are reading and
  working on.
`scan_size`
  In the GUI, we generally support visualizing data containing rectangular 2D scans. For
  some data set types, you can specify a scan_size as a tuple `(y, x)`; for
  others it is even required. When using the Python API, you are free to use n-dimensional
  `scan_size`, if the chosen analysis supports it.

Supported formats
~~~~~~~~~~~~~~~~~

LiberTEM supports the following file formats out of the box, see links for details:

* :ref:`mib`
* :ref:`raw binary`
* :ref:`dm format`
* :ref:`empad`
* :ref:`k2is`
* :ref:`frms6`
* :ref:`blo`
* :ref:`ser`
* :ref:`hdf5`
