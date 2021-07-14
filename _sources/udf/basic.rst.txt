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


.. _`implement udf`:

Implementing a UDF
------------------

The workflow for implementing a UDF starts with subclassing
:class:`~libertem.udf.base.UDF`. In the simplest case, you need to implement the
:meth:`~libertem.udf.base.UDF.get_result_buffers` method and 
:meth:`~libertem.udf.base.UDFFrameMixin.process_frame`.

There are two very common patterns for reductions, reducing over the navigation axes
into a common accumulator for all frames, keeping the shape of a single frame,
or reducing over the signal axes and keeping the navigation axes.

Declaring buffers
~~~~~~~~~~~~~~~~~

A UDF can implement one of these reductions or combinations. To handle indexing
for you, LiberTEM needs to know about the structure of your reduction. You can
build this structure in the :meth:`~libertem.udf.base.UDF.get_result_buffers`
method, by declaring one or more buffers.

These buffers can have a :code:`kind` declared, which corresponds to the two
reduction patterns above: :code:`kind="sig"` for reducing over the navigation
axes (and keeping the signal axes), and :code:`kind="nav"` for reducing over the
signal axes and keeping the navigation axes. There is a third,
:code:`kind="single"`, which allows to declare buffers with custom shapes that
don't correspond directly to the data set's shape.

It is also possible to append additional axes to the buffer's shape using the
:code:`extra_shape` parameter.

:meth:`~libertem.udf.base.UDF.get_result_buffers` should return a :code:`dict` which maps
buffer names to buffer declarations. You can create a buffer declaration by calling
the :meth:`~libertem.udf.base.UDF.buffer` method.

The buffer name is later used to access the buffer via :code:`self.results.<buffername>`,
which returns a view into a NumPy array. For this to work, the name has to be a valid Python
identifier.

Examples of buffer declarations:

.. testcode:: getresultbuffers

   def get_result_buffers(self):
      # Suppose our dataset has the shape (14, 14, 32, 32),
      # where the first two dimensions represent the navigation
      # dimension and the last two dimensions represent the signal
      # dimension.

      buffers = {
         # same shape as navigation dimensions of dataset, plus two
         # extra dimensions of shape (3, 2). The full shape is
         # (14, 14, 3, 2) in this example. This means this buffer can
         # store an array of shape (3, 2) for each frame in the dataset.
         "nav_buffer": self.buffer(
               kind="nav",
               extra_shape=(3, 2),
               dtype="float32",
         ),

         # same shape as signal dimensions of dataset, plus an extra
         # dimension of shape (2,). Consequently, the full shape is
         # (32, 32, 2) in this example. That means we can store two
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

      return buffers

.. testcleanup:: getresultbuffers

   class TestUDF(UDF):
      def merge(self, dest, src):
         pass

   TestUDF.get_result_buffers = get_result_buffers

   u = TestUDF()
   ctx.run_udf(dataset=dataset, udf=u)

See below for some more real-world examples.

All NumPy dtypes are supported for buffers. That includes the :code:`object`
dtype for arbitrary Python variables. The item just has to be pickleable with
:code:`cloudpickle`.

Note that buffers are only designed to pass lightweight intermediate results
and thus, it is important that the size of the buffer remains small. Having too
large buffers can lead to significant decline in performance.

Implementing the processing function
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now to the actual core of the processing: implementing
:meth:`~libertem.udf.UDFFrameMixin.process_frame`. The method signature looks like this:

.. testcode:: processing

   class ExampleUDF(UDF):
      def process_frame(self, frame):
         ...

.. testcleanup:: processing

   u = ExampleUDF()
   ctx.run_udf(dataset=dataset, udf=u)

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

.. _`sumsig`:
   
.. testcode:: sumsig

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
         Sum up all pixels in this frame and store the result in the
         `pixelsum` buffer. `self.results.pixelsum` is a view into the
         result buffer we defined above, and corresponds to the entry
         for the current frame we work on. We don't have to take care
         of finding the correct index for the frame we are processing
         ourselves.
         """
         self.results.pixelsum[:] = np.sum(frame)

   res = ctx.run_udf(
      udf=SumOverSig(),
      dataset=dataset,
   )

   # to access the named buffer as a NumPy array:
   res['pixelsum'].data

On a 4D data set, this operation is roughly equivalent to :code:`np.sum(arr, axis=(2, 3))`.

Merging partial results
~~~~~~~~~~~~~~~~~~~~~~~

As :ref:`described above <how UDFs work>`, data from multiple partitions is
processed in parallel. That also means that we need a way of merging partial
results into the final result. In the example above, we didn't need to do anything:
we only have a :code:`kind="nav"` buffer, where merging just means assigning the
result of one partition to the right slice in the final result. This is done by
the default implementation of :meth:`~libertem.udf.base.UDF.merge`. 

In case of :code:`kind="sig"` buffers and the corresponding reduction, assignment would
just overwrite the result from the previous partition with the one from the current partition,
and is not the correct operation. So let's have a look at the merge method:

.. testcode:: merge

   class ExampleUDF(UDF):
      def merge(self, dest, src):
         pass

.. testcleanup:: merge

   u = ExampleUDF()
   ctx.run_udf(dataset=dataset, udf=u)

:code:`dest` is the result of all previous merge calls, and :code:`src` is the
result from a single new partition. Your :code:`merge` implementation should read from both
:code:`dest` and :code:`src` and write the result back to :code:`dest`.

Here is an example demonstrating :code:`kind="sig"` buffers and the :code:`merge` function:
   
.. testcode:: realmerge

   import numpy as np
   from libertem.udf import UDF


   class MaxUDF(UDF):
      def get_result_buffers(self):
         """
         Describe the buffers we need to store our results:
         kind="sig" means we want to have a value for each coordinate
         in the signal dimensions (i.e. a value for each pixel of the
         diffraction patterns). We name our buffer 'maxbuf'.
         """
         return {
            'maxbuf': self.buffer(
               kind="sig", dtype=self.meta.input_dtype
            )
         }

      def preprocess(self):
         """
         Initialize buffer with neutral element for maximum.
         """
         self.results.maxbuf[:] = -np.inf

      def process_frame(self, frame):
         """
         In this function, we have a frame and the buffer `maxbuf`
         available, which we declared above. This function is called
         for all frames / diffraction patterns in the data set. The
         maxbuf is a partial result, and all partial results will
         later be merged (see below).

         In this case, we determine the maximum from the current
         maximum and the current frame, for each pixel in the
         diffraction pattern.

         Notes:

         - You cannot rely on any particular order of frames this function
           is called in.
         - Your function should be pure, that is, it should not have side
           effects beyond modifying the content of result buffers or task data,
           and should only depend on it's input parameters, including
           the UDF object :code:`self`.
         """
         self.results.maxbuf[:] = np.maximum(frame, self.results.maxbuf)

      def merge(self, dest, src):
         """
         merge two partial results, from src into dest
         """
         dest.maxbuf[:] = np.maximum(dest.maxbuf, src.maxbuf)

   res = ctx.run_udf(
      udf=MaxUDF(),
      dataset=dataset,
   )

   # to access the named buffer as a NumPy array:
   res['maxbuf'].data


For more complete examples, you can also have a look at the functions
implemented in the sub-modules of :code:`libertem.udf` and at our :ref:`packages`.

Passing parameters
~~~~~~~~~~~~~~~~~~

By default, keyword arguments that are passed to the constructor of a UDF are
available as properties of :code:`self.params`. This allows clean handling and passing
of parameters in distributed execution scenarios, see below.

.. testsetup:: params

   def correlate_peaks(frame, peaks):
      pass

   peaks = None

.. testcode:: params

    class MyUDF(UDF):

        def process_frame(self, frame):
            result = correlate_peaks(frame, self.params.peaks)
            ...

    udf = MyUDF(peaks=peaks, other=...)

.. testcleanup:: params

   def get_result_buffers():
      return {}

   udf.get_result_buffers = get_result_buffers

   ctx.run_udf(dataset=dataset, udf=udf)

Declaring a constructor
~~~~~~~~~~~~~~~~~~~~~~~

In order to document parameters of an UDF and avoid typos with parameter names,
it is good practice to define a constructor for any UDF that will be re-used and
shared. Since functions of an UDF are executed both on the main node and on
worker processes, UDFs will be pickled and sent over the network in the process.
To avoid transferring unwanted state or unnecessary or unpicklable member
variables, a clean copy of an UDF is created before transfer. This clean copy is
created by re-instantiating the UDF class with the parameters that were passed
to the constructor of the UDF base class.

That means a user-defined constructor has to fulfill two conditions:

1. It has to pass any parameters to the superclass constructor.
2. It has to accept exactly the parameters that it passed to the superclass
   constructor whenever a clean copy is created and behave the same way as in the
   original call.

That means modification of parameters other than assigning values for default parameters
can lead to complications and should be avoided, if possible.

.. testcode:: constructor

    class MyParamUDF(UDF):
        '''
        This UDF demonstrates how to define a constructor for a UDF.

        Parameters
        ----------
        my_param
            A parameter that is passed into the UDF for demonstration purposes.
            It is mirrored in the `demo` result buffer.
        '''
        def __init__(self, my_param=None):
            # Assigning a default parameter works,
            # provided the UDF accepts it in subsequent calls
            # exactly as it passed it to the superclass:
            if my_param is None:
                my_param = "Eins, zwei drei!"
            
            # !!!DON'T!!! self.my_param = my_param
            # !!!DON'T!!! super().__init__()
            # We have to pass it to the superclass constructor instead.
            # This makes sure it will be available via self.params and
            # on clean copies.

            # !!!DON'T!!! super().__init__(other_param=my_param)
            # This would trigger a TypeError when
            # MyParamUDF(other_param=my_param) is called for a clean copy.

            # DO pass all parameters to the superclass exactly as this
            # class would accept them
            super().__init__(my_param=my_param)

        # The rest of this UDF just mirrors the parameter
        # back in a result buffer for demonstration.

        def get_result_buffers(self):
            '''
            Declare a result buffer for the parameter
            '''
            return {
                'demo': self.buffer(
                    kind='single',
                    dtype=object
                ),
            }

        def process_frame(self, frame):
            '''
            We assign the parameter to the result buffer
            '''
            self.results.demo[:] = self.params.my_param
            
        def merge(self, dest, src):
            '''
            We pass through the result buffer
            '''
            dest.demo[:] = src.demo


    res = ctx.run_udf(dataset=dataset, udf=MyParamUDF())
    assert res['demo'].data[0] == "Eins, zwei drei!"

    res2 = ctx.run_udf(dataset=dataset, udf=MyParamUDF(my_param=42))
    assert res2['demo'].data[0] == 42


Initializing result buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To allow a UDF to initialize a result buffer to the correct values,
the method :meth:`~libertem.udf.base.UDFPreprocessMixin.preprocess`
can be implemented. It is run once per partition and assigning to
:code:`kind="nav"` result buffers will assign to the results of the
whole partition. See :code:`MaxUDF` above for an example.

.. _`progress bar`:

Running UDFs
------------

As shown in the examples above, the :meth:`~libertem.api.Context.run_udf` method
of :class:`~libertem.api.Context` is used to run UDFs. Usually, you only need to
pass an instance of your UDF and the dataset you want to run on:
   
.. testcode:: run

    udf = YourUDF(param1="value1")
    res = ctx.run_udf(udf=udf, dataset=dataset)

You can also enable a progress bar:

.. testcode:: run

    udf = YourUDF(param1="value1")
    res = ctx.run_udf(udf=udf, dataset=dataset, progress=True)

:meth:`~libertem.api.Context.run_udf` returns a :code:`dict`, having the buffer
names as keys (as defined in :meth:`~libertem.udf.UDF.get_result_buffers`) and
:class:`~libertem.common.buffers.BufferWrapper` instances as values. You
can use these in any place you would use a NumPy array, for example as an argument to
NumPy functions, or you can explicitly convert them to NumPy arrays by accessing
the :code:`.data` attribute, or by calling :meth:`numpy.array`:
   
.. testcode:: run

   import numpy as np

   res = ctx.run_udf(udf=udf, dataset=dataset)
   # convert to NumPy array, assuming we declared a buffer
   # with name `buf1`:
   arr = res['buf1'].data
   arr = np.array(res['buf1'])

   # or directly treat as array:
   np.sum(res['buf1'])

If you want to run multiple independent UDFs on a single :code:`DataSet`,
you can pass in a list of UDFs to :meth:`~libertem.api.Context.run_udf`. This can be faster
than making two passes over the whole :code:`DataSet`:

.. testcode:: run

   from libertem.udf.sum import SumUDF

   # results are returned as a tuple, so we can unpack them here into
   # `res` and `res_sum`:
   res, res_sum = ctx.run_udf(udf=[udf, SumUDF()], dataset=dataset)


.. _`udf roi`:

Regions of interest
~~~~~~~~~~~~~~~~~~~

In addition, you can pass the :code:`roi` (region of interest) parameter, to
run your UDF on a selected subset of data. :code:`roi` should be a NumPy array
containing a bool mask, having the shape of the navigation axes of the dataset.
For example, to process a random subset of a 4D-STEM dataset:
   
.. testcode:: run

   import numpy as np

   # If your dataset has shape `(14, 14, 32, 32)` with two signal
   # and two navigation dimensions, `dataset.shape.nav`
   # translates to `(14, 14)`.
   roi = np.random.choice(a=[False, True], size=dataset.shape.nav)
   ctx.run_udf(udf=udf, dataset=dataset, roi=roi)

Note that the result array only contains values for the selected indices, all
other indices are set to :code:`nan` (or, if the dtype doesn't support nan,
some other, not further defined value). It is best to limit further processing
to the same :code:`roi`.

You can also access a flat array that is not filled up with :code:`nan` using
:code:`.raw_data`:

.. testcode:: run

   res = ctx.run_udf(udf=udf, dataset=dataset, roi=roi)
   res['buf1'].raw_data

.. _plotting:

Live Plotting
-------------

.. versionadded:: 0.7.0

LiberTEM can display a live plot of the UDF results. In the most simple case,
this can be done by setting :code:`plots=True` in
:meth:`~libertem.api.Context.run_udf`. 

.. testsetup:: live

    from libertem.udf.sum import SumUDF
    udf = SumUDF()

.. testcode:: live

    ctx.run_udf(dataset=dataset, udf=udf, plots=True)

See the following items for a full demonstration, including setting up fully
customized plots. The API reference can be found in :ref:`viz reference`.

.. toctree::

    liveplotting

.. _partial:

Partial results
---------------

.. versionadded:: 0.7.0

Instead of only getting the whole result after the UDF has finished running, you
can also use :meth:`~libertem.api.Context.run_udf_iter` to get a generator for
partial results:

.. testsetup:: partial

    from libertem.udf.sum import SumUDF


.. testcode:: partial

    for udf_results in ctx.run_udf_iter(dataset=dataset, udf=SumUDF()):
        # ... do something interesting with `udf_results`:
        a = np.sum(udf_results.buffers[0]['intensity'])

    # after the loop, `udf_results` contains the final results as usual

While the UDF execution is running, the UDF object should not be modified since
that leads to undefined behavior. In particular, nested or concurrent execution
of the same UDF objects must be avoided since it modifies the buffers that are
allocated internally while a UDF is running.

Asynchronous execution
----------------------

It is also possible to integrate LiberTEM into an async script or application by
passing :code:`sync=False` to :meth:`~libertem.api.Context.run_udf_iter` or
:meth:`~libertem.api.Context.run_udf`:

.. Not run with docs-check since we can't easily test async code there...

.. code-block:: python

    async for udf_results in ctx.run_udf_iter(dataset=dataset, udf=udf, sync=False):
        # ... do something interesting with `udf_results`:
        a = np.sum(udf_results[0]['intensity'])

    # or the version without intermediate results:
    udf_results = await ctx.run_udf(dataset=dataset, udf=udf, sync=False)

See the items below for a more comprehensive demonstration and documentation:

.. toctree::

    async