.. _`loading data`:

Loading data
============

To efficiently handle files larger than main memory, LiberTEM never loads the
whole data set at once. Calling the :meth:`~libertem.api.Context.load`
function only checks that the dataset exists and is value before providing Python
with an object which can be used in later computation. Running an analysis
on this object with :meth:`~libertem.api.Context.run` or
:meth:`~libertem.api.Context.run_udf` then streams the data from mass storage
in optimal-sized chunks, such that even very large datasets can be processed without
saturating the system resources.

See :ref:`sample data` for publicly available datasets for testing.

There are two main ways of opening a data set in LiberTEM: using the GUI, or the
Python API.

Loading through the API
~~~~~~~~~~~~~~~~~~~~~~~

In the API, you can use :meth:`libertem.api.Context.load`. The general
pattern is:

.. code-block:: python

   ctx = Context()
   ctx.load("typename", path="/path/to/some/file", arg1="val1", arg2=42)

So, you need to specify the data set type, the path, and dataset-specific
arguments. These arguments are documented below.

For most file types, it is possible to automatically detect the type and
parameters, which you can trigger by using :code:`"auto"` as file type:

.. code-block:: python

   ctx.load("auto", path="/path/to/some/file")

For the full list of supported file formats with links to their reference
documentation, see :ref:`supported formats` below.

.. _`Loading using the GUI`:

Loading using the GUI
~~~~~~~~~~~~~~~~~~~~~

Using the GUI, mostly the same parameters need to be specified, although some
are only available in the Python API. Tuples (for example for :code:`nav_shape`)
have to be entered as separated values into the fields. You can hit a comma to jump to
the next field. We follow the NumPy convention here and specify the "fast-access" dimension
last, so a value of :code:`42`, :code:`21` would mean the same as specifying
:code:`(42, 21)` in the Python API, setting :code:`y=42` and :code:`x=21`.

See the :ref:`GUI usage page <usage documentation>` for more information on the GUI. 

For more general information about how LiberTEM structures data see :ref:`the concepts section <concepts>`.

Common parameters
~~~~~~~~~~~~~~~~~

There are some common parameters across data set types:

`name`
  The name of the data set, for display purposes. Only used in the GUI.
`nav_shape`
  In the GUI, we generally support visualizing data containing rectangular 2D scans. For
  all the dataset types, you can specify a nav_shape as a tuple `(y, x)`. If the dataset
  isn't 4D, the GUI can reshape it to 4D. When using the Python API, you are free to
  use n-dimensional `nav_shape`, if the data set and chosen analysis supports it.
`sig_shape`
  In the GUI, you can specify shape of the detector as :code:`height`, :code:`width`, but
  when using the Python API, it can be of any dimensionality.
`sync_offset`
  You can specify a `sync_offset` to handle synchronization or acquisition problems.
  If it's positive, `sync_offset` number of frames will be skipped from the start of the input data.
  If it's negative, the dataset will be padded by `abs(sync_offset)` number of frames at the beginning.
`io_backend`
  Different methods for I/O are available in LiberTEM, which can influence performance. 
  See :ref:`io backends` for details.

.. note::
  When using :code:`sync_offset` or a :code:`nav_shape` that exceeds the size of the input data
  it is currently not well-defined if zero-filled frames are to be generated or if the missing data is skipped.
  Most dataset implementations seem to skip the data. See :issue:`1384` for discussion, feedback welcome!

.. _`supported formats`:

Supported formats
~~~~~~~~~~~~~~~~~

LiberTEM supports the following file formats out of the box, see links for details:

* :ref:`mib`
* :ref:`raw binary`
* :ref:`raw csr`
* :ref:`npy format`
* :ref:`dm format`
* :ref:`empad`
* :ref:`k2is`
* :ref:`frms6`
* :ref:`blo`
* :ref:`ser`
* :ref:`hdf5`
* :ref:`seq`
* :ref:`mrc`
* :ref:`tvips`

Furthermore, two alternative mechanisms exist for interfacing LiberTEM with data loaded
elsewhere in Python via other libraries:

- a memory data set can be constructed from a NumPy array for testing
  purposes. See :ref:`memory` for details.
- a Dask data set can be constructed from a Dask array. Depending on the
  method used to construct the source array this can achieve good performance.
  See :ref:`daskds` for details.

.. _`data conversion`:

Dataset conversion
~~~~~~~~~~~~~~~~~~

LiberTEM supports a mechanism to efficiently convert any supported dataset 
into a Numpy binary file (:code:`.npy`), which can then be loaded into memory
independently of LiberTEM (or read as a :code:`npy` format dataset as above).

.. versionadded:: 0.12.0

To convert a dataset to npy, use the :meth:`~libertem.api.Context.export_dataset` method:

.. code-block:: python

   with lt.Context() as ctx:
       ctx.export_dataset(dataset, './output_path.npy')


As of this time only exporting to the :code:`npy` format is supported, but other formats would be
possible as the need arose.

Alternatively, you can create Dask arrays from LiberTEM datasets via the :ref:`Dask integration <daskarray>`.
These arrays can then be stored with
`Dask's built-in functions <https://docs.dask.org/en/stable/array-creation.html#store-dask-arrays>`_
or through additional libraries such as `RosettaSciIO <https://rosettasciio.readthedocs.io/en/latest/index.html>`_.
