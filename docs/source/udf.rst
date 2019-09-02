.. _`user-defined functions`:

User-defined functions
======================

A common case for analyzing big EM data sets is running a reduction operation
on each individual detector frame or other small subsets of a data set and then
combining the results of these reductions to form the complete result. This should
cover a wide range of use cases, from simple mathematical operations (for
example statistics) to complex image processing and analysis, like feature extraction.

The user-defined functions (UDF) interface of LiberTEM allows users to run their
own reduction functions easily, without having to worry about parallelizing,
I/O, the details of buffer management and so on. This corresponds to
a simplified `MapReduce programming model <https://en.wikipedia.org/wiki/MapReduce>`_,
where the intermediate re-keying and shuffling step is omitted.

It can be helpful to review :doc:`some general concepts <concepts>` before
reading the following sections.

Getting started
---------------

The easiest way of running a function over your data is using the 
:meth:`~libertem.api.Context.map` method of the LiberTEM API.

For example, to calculate the sum over the last signal axis:

.. code-block:: python

      >>>result = ctx.map(
      >>>      dataset=dataset,
      >>>      f=functools.partial(np.sum, axis=-1)
      >>>)
      >>>result

The function specified via the :code:`f` parameter is called for each frame / diffraction pattern.

See `Auto UDF`_ below for details. This is most suited for simple functions; once you have
parameters or want to re-use some data across function calls, you should create a
:class:`~libertem.udf.UDF` subclass instead.

Read on for a more in-detail view of UDFs and their capabilities.

How UDFs works
--------------

.. image:: ./images/udf-diagram.png

To allow for parallel processing, data is first divided into partitions along the navigation axes,
which are worked on by different worker processes. Then, for each frame of a partition, a
user-defined function :meth:`~libertem.udf.UDFFrameMixin.process_frame` is called,
which is free to do any imaginable processing.

As a result of splitting the data set into partitions, the results then need to be merged
back together. This is accomplished by calling the :meth:`~libertem.udf.UDF.merge` method
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
potentially :doc:`on multiple computers <architecture>`. The loop over individual frames is
run in the worker processes, and the merge function is run in the main process, accumulating the
results, every time the results for a partition are available. 

In addition to :meth:`~libertem.udf.UDFFrameMixin.process_frame`, there are two more methods
available for overriding, to work on larger units of data at the same time:
:meth:`~libertem.udf.UDFTileMixin.process_tile`
and :meth:`~libertem.udf.UDFPartitionMixin.process_partition`. They can be used for optimizing
some operations, and are documented in the :doc:`advanced topics <udf/advanced>` section.

Implementing a UDF
------------------

The workflow for implementing a UDF starts with subclassing
:class:`~libertem.udf.UDF`. In the simplest case, you need to implement the
:meth:`~libertem.udf.UDF.get_result_buffers` method and 
:meth:`~libertem.udf.UDFFrameMixin.process_frame`.

There are two very common patterns for reductions, either reducing over the navigation axes
into a common accumulator for all frames, keeping the shape of a single frame,
or reducing over the signal axes and keeping the navigation axes.

A UDF can implement one of these reductions, or even combinations. To handle indexing for you,
LiberTEM needs to know about the sturcture of your reduction. You can build this structure in the
:meth:`~libertem.udf.UDF.get_result_buffers` method, by declaring one or more buffers.

Declaring buffers
~~~~~~~~~~~~~~~~~

These buffers can have a :code:`kind` declared, which corresponds to the two reduction patterns above:
:code:`kind="sig"` for reducing over the navigation axes (and keeping the signal axes), and 
:code:`kind="nav"` for reducing over the signal axes and keeping the navigation axes. There is a
third, :code:`kind="single"`, which stores just a single value.

It is also possible to append new axes to the end of the buffer using the
:code:`extra_shape` parameter.

:meth:`~libertem.udf.UDF.get_result_buffers` should return a :code:`dict` which maps
buffer names to buffer declarations. You can create a buffer declaration by calling
the :meth:`~libertem.udf.UDF.buffer` method.

The buffer name is later used to access the buffer via :code:`self.results.<buffername>`,
which returns a view into a numpy array. For this to work, the name has to be a valid Python
identifier.

Examples of buffer declarations (this is a :code:`dict` as it would be returned by
:meth:`~libertem.udf.UDF.get_result_buffers`):

.. code-block:: python

   # Suppose our dataset has the shape (14, 14, 32, 32),
   # where the first two dimensions represent the navigation
   # dimension and the last two dimensions represent the signal dimension.

   buffers = {
      # same shape as navigation dimensions of dataset, plus two extra dimensions of
      # shape (3, 2). The full shape is (14, 14, 3, 2) in this example.
      # This means this buffer can store an array of shape (3, 2) for each frame in the dataset.
      "nav_buffer": self.buffer(
          kind="nav",
          extra_shape=(3, 2),
          dtype="float32",
      ),

      # same shape as signal dimensions of dataset, plus an extra dimension of shape (2,).
      # so the full shape is (32, 32, 2) in this example. That means we can store two
      # float32 values for each pixel of the signal dimensions.
      "sig_buffer": self.buffer(
          kind="sig",
          extra_shape=(2,),
          dtype="float32",
      ),

      # buffer of shape (16, 16); shape is unrelated to dataset shape
      "single_buffer": self.buffer(
          kind="single",
          extra_shape=(16, 16),
          dtype="float32",
      ),

   }

See below for some more real-world examples.

All numpy dtypes are supported for buffers. That includes the :code:`object`
dtype for arbitrary Python variables. The item just has to be pickleable with
:code:`cloudpickle`.

Note that buffers are only designed to pass lightweight intermediate results
and thus, it is important that the size of the buffer remains small. Having too
large buffers can lead to significant decline in performance.

Implementing the processing function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now to the actual "meat" of the processing: implementing
:meth:`~libertem.udf.UDFFrameMixin.process_frame`. The method signature looks like this:

.. code-block:: python

   class YourUDF(UDF):
      def process_frame(self, frame):
         pass

The general idea is that you get a single frame from the data set, do your processing,
and write the results to one of the previously declared buffers, via :code:`self.results.<buffername>`.
When accessing a :code:`kind="nav"` buffer this way, you automatically get a view into the buffer
that corresponds to the current frame that is being processed. In case of :code:`kind="sig"`
or :code:`kind="single"`, you get the whole buffer.

Intuitively, with :code:`kind="sig"` (and :code:`kind="single"`), you are most
likely implementing an operation like :code:`buf = f(buf, frame)`. That is, you
are computing a new result based on a single (new) frame and the results from all
previous frames, and overwrite the results with the new value(s).

With :code:`kind="nav"`, you compute independent results for each frame,
which are written to different positions in the result buffer. Because of the independence
between frames, you don't need to merge with a previous value; the result is simply written
to the correct index in the result buffer (via the aforementioned view).

As an easy example, let's have a look at a function that simply sums up each frame
to a single value. This is a :code:`kind="nav"` reduction, as we sum over all values
in the signal dimensions:

.. testsetup:: *

    from libertem import api
    from libertem.executor.inline import InlineJobExecutor

    ctx = api.Context(executor=InlineJobExecutor())
    dataset = ctx.load("memory", datashape=(16, 16, 16), sig_dims=2)
   
.. testcode::

   import numpy as np   
   from libertem.udf import UDF


   class SumOverSig(UDF):
      def get_result_buffers(self):
         """
         Describe the buffers we need to store our results:
         kind="nav" means we want to have a value for each coordinate
         in the navigation dimensions. We name our buffer 'pixelsum'.
         """
         return {
            'pixelsum': self.buffer(
               kind="nav", dtype="float32"
            )
         }

      def process_frame(self, frame):
         """
         Sum up all pixels in this frame and store the result in the `pixelsum`
         buffer. `self.results.pixelsum` is a view into the result buffer we
         defined above, and corresponds to the entry for the current frame we
         work on. We don't have to take care of finding the correct index for
         the frame we are processing ourselves.
         """
         self.results.pixelsum[:] = np.sum(frame)

   res = ctx.run_udf(
      udf=SumOverSig(),
      dataset=dataset,
   )

   print(res['pixelsum'].shape)


Here is another example, demonstrating :code:`kind="sig"` buffers and the merge function:


.. include:: udf/max.py
   :code:


For a more complete example, please have a look at the functions implemented in the sub-modules of :code:`libertem.udf`,
for example :code:`libertem.udf.blobfinder`.

Passing parameters
------------------

By default, keyword arguments that are passed to the constructor of a UDF are available as properties of :code:`self.params`:

.. code-block:: python

    class MyUDF(UDF):

        def process_frame(self, frame):
            result = correlate_peaks(frame, self.params.peaks)
            ...

    udf = MyUDF(peaks=peaks, ...)


AUX data
~~~~~~~~

If a parameter is an instance of :class:`~libertem.common.buffers.BufferWrapper`, the UDF interface will interpret it as auxiliary data. It will set the views for each tile/frame/partition accordingly so that accessing the parameter returns a view of the auxiliary data matching the data portion that is currently being processed. That way, it is possible to pass parameters individually for each frame or to mask the signal dimension. The :meth:`~libertem.udf.UDF.aux_data` class method helps to wrap data into a suitable :class:`~libertem.common.buffers.BufferWrapper`.

For masks in the signal dimension that are used for dot products in combination with per-tile processing, a :class:`~libertem.job.masks.MaskContainer` allows to use more advanced slicing and transformation methods targeted at preparing mask stacks for optimal dot product performance.

Task data
---------

A UDF can generate task-specific intermediate data on the worker nodes by defining a :meth:`~libertem.udf.UDF.get_task_data` method. The result is available as an instance of :class:`~libertem.udf.UDFData` in :code:`self.task_data`. Depending on the circumstances, this can be more efficient than making the data available as a parameter since it avoids pickling, network transport and unpickling.

This non-trivial example from :class:`~libertem.udf.blobfinder.SparseCorrelationUDF` creates a :class:`~libertem.job.masks.MaskContainer` based on the parameters in :code:`self.params`. This :class:`~libertem.job.masks.MaskContainer` is then available as :code:`self.task_data['mask_container']` within the processing functions.

.. code-block:: python

    def get_task_data(self):
        mask = mask_maker(self.params)
        crop_size = mask.get_crop_size()
        size = (2 * crop_size + 1, 2 * crop_size + 1)
        template = mask.get_mask(sig_shape=size)
        steps = self.params.steps
        peak_offsetY, peak_offsetX = np.mgrid[-steps:steps + 1, -steps:steps + 1]

        offsetY = self.params.peaks[:, 0, np.newaxis, np.newaxis] + peak_offsetY - crop_size
        offsetX = self.params.peaks[:, 1, np.newaxis, np.newaxis] + peak_offsetX - crop_size

        offsetY = offsetY.flatten()
        offsetX = offsetX.flatten()

        stack = functools.partial(
            sparse_template_multi_stack,
            mask_index=range(len(offsetY)),
            offsetX=offsetX,
            offsetY=offsetY,
            template=template,
            imageSizeX=self.meta.dataset_shape.sig[1],
            imageSizeY=self.meta.dataset_shape.sig[0]
        )
        # CSC matrices in combination with transposed data are fastest
        container = MaskContainer(mask_factories=stack, dtype=np.float32,
            use_sparse='scipy.sparse.csc')

        kwargs = {
            'mask_container': container,
            'crop_size': crop_size,
        }
        return kwargs


Auto UDF
--------

The :class:`~libertem.udf.AutoUDF` class and :meth:`~libertem.api.Context.map` method allow to run simple functions that accept a frame as the only parameter with an auto-generated :code:`kind="nav"` result buffer over a dataset ad-hoc without defining an UDF class. For more advanced processing, such as custom merge functions, post-processing or performance optimization through tiled processing, defining an UDF class is required.

As an alternative to Auto UDF, you can use the :meth:`~libertem.contrib.dask.make_dask_array` method to create a `dask.array <https://docs.dask.org/en/latest/array.html>`_ from a :class:`~libertem.io.dataset.base.DataSet` to perform calculations. See :ref:`Integration with Dask arrays<daskarray>` for more details.

The :class:`~libertem.udf.AutoUDF` class determines the output shape and type by calling the function with a mock-up frame of the same type and shape as a real detector frame and converting the return value to a numpy array. The :code:`extra_shape` and :code:`dtype` parameters for the result buffer are derived automatically from this numpy array.

Additional constant parameters can be passed to the function via :meth:`functools.partial`, for example. The return value should be much smaller than the input size for this to work efficiently.

Example: Calculate sum over the last signal axis.

.. code-block:: python

      result = ctx.map(
            dataset=dataset,
            f=functools.partial(np.sum, axis=-1)
      )

      # or alternatively:

      udf = AutoUDF(f=functools.partial(np.sum, axis=-1))
      result = self.run_udf(dataset=dataset, udf=udf)

.. _tiled:


Debugging
---------

See also the :ref:`general section on debugging <debugging udfs>`.

TODO: document common pitfalls here.

See also
--------

.. toctree::
   concepts
   udf/reference
   udf/advanced
