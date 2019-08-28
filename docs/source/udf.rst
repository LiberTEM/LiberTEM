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

It can be helpful to review :doc:`some general concepts <concepts>` before reading this section.

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
run on the worker process, and the merge function is run on the master node, once the results
for a partition are available. 

In addition to :meth:`~libertem.udf.UDFFrameMixin.process_frame`, there are two more methods
available for overriding, to work on larger units of data at the same time:
:meth:`~libertem.udf.UDFTileMixin.process_tile`
and :meth:`~libertem.udf.UDFPartitionMixin.process_partition`. They can be used for optimizing
some operations, and are documented in the :doc:`advanced topics <udf/advanced>` section.


Declaring Buffers
-----------------

There are two very common patterns for reductions, either reducing over the navigation axes
into a common accumulator for all frames, or reducing over the signal axes and keeping the
navigation axes.

To make your UDF code more readable, LiberTEM supports these cases out of the box: you can
declare buffers in the :meth:`~libertem.udf.UDF.get_result_buffers` method.
These buffers can have a :code:`kind` declared, which corresponds to the two reduction patterns above:
:code:`kind="sig"` for reducing over the navigation axes (and keeping the signal axes), and 
:code:`kind="nav"` for reducing over the signal axes and keeping the navigation axes. There is a
third, :code:`kind="single"`, which stores just a single value. It is also possible to append
new axes to the end of the buffer using the :code:`extra_shape` parameter.

By keeping with these patterns, a UDF author doesn't have to take care of indexing, and code 
can easily be written to support any dimensionality in both navigation and signal axes.

# XXX FIXME XXX: rewrite the following parts
================

In the UDF interface of LiberTEM, buffers are the tools to save and pass on the
intermediate results of computation. Currently, LiberTEM supports three different
types of buffer: :code:`"sig"`, :code:`"nav"`, and :code:`"single"`. By setting :code:`kind="sig"`, users
can make the buffer to have the same dimension as the signal dimension. By setting
the :code:`kind="nav"`, users can make the buffer to have the same dimension as the navigation
dimension. Lastly, by setting :code:`kind="single"`, users can make the buffer to have an arbitrary 
dimension of their choice. Note that in the case of :code:`"single"` buffer, users may specify 
the dimension of the buffer through :code:`extra_shape` parameter. If :code:`extra_shape` 
parameter is not specified, the buffer is assumed to have :code:`(1,)` dimension. Additionally, 
users may also specify :code:`extra_shape` parameters for :code:`"sig"` or :code:`"nav"` buffers. 
In that case, the dimensions specified by :code:`extra_shape` parameter will be added to the 
dimension of :code:`dataset.shape.sig` or :code:`dataset.shape.nav`, with respect to each component. As an example,
one may specify the buffers as following:

.. include:: udf/buffer_types.py
   :code:

One can access a buffer of interest via :code:`self.results.<buffername>`, from which one can get a view into a numpy array
that the buffer is storing. This numpy array corresponds to the current intermediate result that LiberTEM is working
on and can be intermediate results of processing frames/tiles/partitions. 
Note that buffers are only designed to pass lightweight intermediate results and thus, it is important
that the size of the buffer remains small. Otherwise, it could lead to significant decline in performance.

All numpy dtypes are supported for buffers. That includes the :code:`object` dtype for arbitrary Python variables. The item just has to be pickleable with :code:`cloudpickle`.

By-frame processing
-------------------
Note that :meth:`~libertem.udf.UDFFrameMixin.process_frame` method can interpreted in a slightly different manner for different types of buffer with which you
are dealing. If the type of the buffer is :code:`"sig"`, then :meth:`~libertem.udf.UDFFrameMixin.process_frame` can be viewed as iteratively `merging` the previous
computations (i.e., the result computed on previously considered set of frames) and a newly added frame. If the type of
the buffer is :code:`"nav"`, then :meth:`~libertem.udf.UDFFrameMixin.process_frame` can be viewed as performing operations on each frame independently. Intuitively, when the type of the buffer is :code:`"nav"`, which means that it uses the navigation dimension, two different frames
correspond to two different scan positions, so the `merging` is in fact an assignment of the result to the correct slot in the result buffer. Lastly, if the type of the buffer is :code:`"single"`, then :meth:`~libertem.udf.UDFFrameMixin.process_frame` can be
interpreted in either way.

As an easy example, let's have a look at a function that simply sums up each frame
to a single value:


.. include:: udf/sumsig.py
   :code:


Here is another example, demonstrating :code:`kind="sig"` buffers and the merge function:


.. include:: udf/max.py
   :code:


For a more complete example, please have a look at the functions implemented in the sub-modules of :code:`libertem.udf`,
for example :code:`libertem.udf.blobfinder`.

Auto UDF
--------

The :class:`~libertem.udf.AutoUDF` class and :meth:`~libertem.api.Context.map` method allow to run simple functions that accept a frame as the only parameter with an auto-generated :code:`kind="nav"` result buffer over a dataset ad-hoc without defining an UDF class. For more advanced processing, such as custom merge functions, post-processing or performance optimization through tiled processing, defining an UDF class is required.

As an alternative to Auto UDF, you can use the :meth:`~libertem.contrib.dask.make_dask_array` method to create a `dask.array <https://docs.dask.org/en/latest/array.html>`_ from a :class:`~libertem.io.dataset.base.DataSet` to perform calculations. See :ref:`Integration with Dask arrays<daskarray>` for more details.

The :class:`~libertem.udf.AutoUDF` class determines the output shape and type by calling the function with a mock-up frame of the same type and shape as a real detector frame and converting the return value to a numpy array. The :code:`extra_shape` and :code:`dtype` parameters for the result buffer are derived automatically from this numpy array. Additional constant parameters can be passed to the function via :meth:`functools.partial`, for example. The return value should be much smaller than the input size for this to work efficiently.

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
