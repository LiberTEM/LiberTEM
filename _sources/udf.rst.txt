.. _`user-defined functions`:

User-defined functions (UDFs)
=============================

.. testsetup:: *

   import numpy as np
   from libertem import api
   from libertem.executor.inline import InlineJobExecutor

   from libertem.udf import UDF as RealUDF

   # We override UDF in such a way that it can be used
   # without implementing all methods
   class UDF(RealUDF):
      def get_result_buffers(self):
         return {}

      def process_frame(self, frame):
         pass

   class YourUDF(RealUDF):
      def get_result_buffers(self):
         return {'buf1': self.buffer(kind="nav")}

      def process_frame(self, frame):
         self.results.buf1[:] = 42

   ctx = api.Context(executor=InlineJobExecutor())
   data = np.random.random((16, 16, 32, 32)).astype(np.float32)
   dataset = ctx.load("memory", data=data, sig_dims=2)
   roi = np.random.choice([True, False], dataset.shape.nav)
   udf = YourUDF()

A common case for analyzing big EM data sets is running a reduction operation
on each individual detector frame or other small subsets of a data set and then
combining the results of these reductions to form the complete result. This should
cover a wide range of use cases, from simple mathematical operations, for
example statistics, to complex image processing and analysis, like feature extraction.

The user-defined functions (UDF) interface of LiberTEM allows you to define and run your
own reduction functions easily, without having to worry about parallelizing,
I/O, or the details of buffer management. This corresponds to
a simplified `MapReduce programming model <https://en.wikipedia.org/wiki/MapReduce>`_,
where the intermediate re-keying and shuffling step is omitted.

LiberTEM ships with some :ref:`utility UDFs <utilify udfs>` that implement
general functionality:

* :ref:`Sum <sum udf>`
* :ref:`Logsum <logsum udf>`
* :ref:`StdDev <stddev udf>`
* :ref:`SumSig <sumsig udf>`
* :ref:`Masks and other linear operations <masks udf>`
* :ref:`Pick <pick udf>`

Also, LiberTEM includes :ref:`ready-to-use application-specific UDFs
<applications>`.

It can be helpful to review :ref:`some general concepts <concepts>` before
reading the following sections.

Getting started
---------------

The easiest way of running a function over your data is using the 
:meth:`~libertem.api.Context.map` method of the LiberTEM API. For example,
to calculate the sum over the last signal axis:

.. testcode:: autoudf

   import functools
   import numpy as np

   result = ctx.map(
      dataset=dataset,
      f=functools.partial(np.sum, axis=-1)
   )
   # access the result as NumPy array:
   np.array(result)
   # or, alternatively:
   result.data

The function specified via the :code:`f` parameter is called for each frame / diffraction pattern.
See :ref:`auto UDF` for more details. This is most suited for simple functions; once you have
parameters or want to re-use some data across function calls, you should create a
:class:`~libertem.udf.base.UDF` subclass instead.

:func:`functools.partial` is a higher-order function that allows to create a new
function by wrapping an existing function and passing additional parameters to
it. In this case, the resulting call to :func:`numpy.sum` within
:code:`ctx.map(...)` is :code:`numpy.sum(frame, axis=-1)`. See
https://docs.python.org/3/library/functools.html#functools.partial for more
details.

Example notebook
----------------

See the following notebook for a demonstration of basic UDF functionality. It
can be downloaded `from our example collection on GitHub
<https://github.com/LiberTEM/LiberTEM/blob/master/examples/Introduction%20to%20UDFs.ipynb>`_.

.. toctree::

   udf/introduction

.. _`how UDFs work`:

How UDFs works
--------------

.. image:: ./images/udf-diagram.png

To allow for parallel processing, data is first divided into partitions along the navigation axes,
which are worked on by different worker processes. Then, for each frame of a partition, a
user-defined function :meth:`~libertem.udf.base.UDFFrameMixin.process_frame` is called,
which is free to do any imaginable processing.

As a result of splitting the data set into partitions, the results then need to be merged
back together. This is accomplished by calling the :meth:`~libertem.udf.base.UDF.merge` method
after all frames of a partition are processed.

In pseudocode, data is processed in the following way:

.. code-block:: python

   result = empty
   for partition in get_partitions(dataset):
      partition_result = empty
      for frame, frame_slice in get_frames(partition):
         frame_result = process_frame(frame)
         partition_result[frame_slice] = frame_result
      merge(dest=result, src=partition_result)

In reality, the loop over partitions is run in parallel using multiple worker processes,
potentially :ref:`on multiple computers <architecture>`. The loop over individual frames is
run in the worker processes, and the merge function is run in the main process, accumulating the
results, every time the results for a partition are available. 

In addition to :meth:`~libertem.udf.base.UDFFrameMixin.process_frame`, there are two more methods
available for overriding, to work on larger/different units of data at the same time:
:meth:`~libertem.udf.base.UDFTileMixin.process_tile`
and :meth:`~libertem.udf.base.UDFPartitionMixin.process_partition`. They can be used for optimizing
some operations, and are documented in the :ref:`advanced topics <advanced udf>` section.

More about UDFs
---------------

Now would be a good time to :ref:`read more about implementing UDFs <implement
udf>` and :ref:`advanced UDF functionality <advanced udf>`. The :ref:`general
section on debugging <debugging udfs>` helps with resolving issues. Once you
have your UDF working, you can proceed to :ref:`UDF profiling <udf profiling>`
to gain insights into the efficiency of your UDF.

.. toctree::

   udf/basic
   udf/advanced
   udf/profiling

.. seealso::

   :ref:`udf reference`
      API documentation for UDFs
