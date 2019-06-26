User-defined functions
======================

A common case for analysing big EM data sets is running a reduction operation
on each frame of the data set. The user-defined functions (UDF) interface
of LiberTEM allows users to run their own reduction functions easily, without having
to worry about parallelizing, I/O, the details of buffer management and so on. 


How UDF works in layman's terms
-----------------------------------

.. image:: ./images/diagram.png

The UDF interface of LiberTEM is heavily utilizing the existing LiberTEM architecture. First,
data is partitioned into several `"partitions"` and distributed across workers. Then, each partition,
which can be viewed as a collection of frames, are processed by the user-defined `process_frame` method.
Here, frames are fetched in an iterative manner, and `process_frame` method performs user-defined operations
on the frames. After all the frames in a partition are processed, LiberTEM iteratively `merges` the results from each worker, which is called
by the `merge` method in UDF class. To summarize, the UDF interface of LiberTEM performs operations at two levels: `process_frame`, which performs user-defined operations
on frames within each partition, and `merge`, which merges the output of `process_frame` from each partition. Note that in both `process_frame` and `merge`, buffers store the intermediate
outcomes of the user-defined operations.

Initializing Buffers
--------------------
In the UDF interface of LiberTEM, buffers are the tools to save and pass on the
intermediate results of computation. Currently, LiberTEM supports three different
types of buffer: `"sig"`, `"nav"`, and `"single"`. By setting `"kind=sig"`, users
can make the buffer to have the same dimension as the signal dimension. By setting
the `"kind=nav"`, users can make the buffer to have the same dimension as the navigation
dimension. Lastly, by setting `"kind=single"`,users can make the buffer to have an arbitrary 
dimension of their choice. Note that in the case of "single" buffer, users may specify 
the dimension of the buffer through `"extra_shape"` parameter. If `"extra_shape"` 
parameter is not specified, the buffer is assumed to have `(1,)` dimension. Additionally, 
users may also specify `"extra_shape"` parameters for `"sig"` or `"nav"` buffers. 
In that case, the dimensions specified by "extra_shape" parameter will be added to the 
dimension of `"sig"` or `"nav"`, with respect to each component. As an example,
one may specify the buffers as following:

.. include:: udf/buffer_types.py
   :code:

One can access a buffer of interest via `self.results.buffername`, from which one can get a view into a numpy array
that the buffer is storing. This numpy array corresponds to the current intermediate result that LiberTEM is working
on and can be intermediate results of processing frames/tiles/partitions. 
Note that buffers are only designed to pass lightweight intermediate results and thus, it is important
that the size of the buffer remains small. Otherwise, it could lead to significant decline in performance.

By-frame processing
-------------------
Note that `process_frame` method can interpreted in a slightly different manner for different types of buffer with which you
are dealing. If the type of the buffer is `"sig"`, then `process_frame` can be viewed as iteratively `merging` the previous
computations (i.e., the result computed on previously considered set of frames) and a newly added frame. If the type of
the buffer is `"nav"`, then `process_frame` can be viewed as performing operations on each frame independently. Intuitively, when the type of the buffer is `nav`, which means that it uses the navigation dimension, two different frames
correspond to two different scan positions so it does not
make much sense to view it as `merging`. Lastly, if the type of the buffer is `"single"`, then `process_frame` can be
interpreted in either way.

As an easy example, let's have a look at a function that simply sums up each frame
to a single value:


.. include:: udf/sumsig.py
   :code:


Here is another example, demonstrating `kind="sig"` buffers and the merge function:


.. include:: udf/max.py
   :code:


For a more complete example, please have a look at the functions implemented in `libertem.udf`,
for example `blobfinder`.


Tiled processing
----------------

Motivation
~~~~~~~~~~

Many operations operations can be optimized by working on stacks of frames. For
example, let's imagine you want to do an elementwise multiplication, which arises
when doing gain correction.  An implementation in pseudo-Python could look like this
(for simplicity, we flatten the frame, weights and result to a 1D array here):

.. code-block:: python

   for idx in range(number_of_pixels):
      result[idx] = frame[idx] * weights[idx]


If you look closely, you may notice that for each frame, all elements from `weights` are accessed.
This is not cache efficient, because you could instead hold on to a single weight value and re-use
it for multiple frames:


.. code-block:: python

   for idx in range(number_of_pixels):
      weight = weights[idx]
      for frame_idx, frame in enumerate(frames):
         result[idx, frame_idx] = frame[idx] * weight

For details, see for example the Wikipedia articles on `Loop nest optimization <https://en.wikipedia.org/wiki/Loop_nest_optimization>`_ and `Locality of reference <https://en.wikipedia.org/wiki/Locality_of_reference>`_.

   
In a real Python implementation, you would of course use numpy with broadcasting,
which takes care of applying the multiplication in an efficient way. But it can only
benefit from the above mentioned re-use if it has data from multiple frames available:

.. code-block:: python
   
   # `tile` and `gain_map` have compatible shapes;
   # tile.shape == (N, Y, X) and gain_map.shape == (Y, X)
   corrected_tile = tile * gain_map


These optimizations are only possible if you have access to data from more than one frame. For
very large frames, another problem arises: a stack of frames would be too large to efficiently handle,
as it would no longer fit into even the L3 cache, which is the largest cache in most CPUs. For these
cases, we support a tiled reading and processing strategy. Tiled means we slice the frame into
disjoint rectangular regions. A tile then is the data from a single rectangular region
for multiple frames.

For example, in case of K2IS data, frames have a shape of `(1860, 2048)`. When reading them
with the tiled strategy, a single tile will contain data from 16 subsequent frames, and each
rectangle has a shape of `(930, 16)` (which happens to be the natural block size for K2IS data).
So the tiles will have a shape of `(16, 930, 16)`, and processing 16 frames from the data set
means reading 256 individual tiles.

Loading a tile of this size as float32 data
still fits comfortably into usual L3 CPU caches (~1MB), and thus enables efficient processing.
As a comparison, a whole `(1860, 2048)` frame is about 15MB large, and accessing it repeatedly
means having to load data from the slower main memory.

Note: you may have noticed that we talk about block sizes of 1MB as efficient in the L3 cache,
but many CPUs have larger L3 caches. As the L3 cache is shared between cores, and LiberTEM tries
to use multiple cores, the effectively available L3 cache has to be divided by number of cores.

TODO: documentation on implementing `process_tile`

Debugging
---------

TODO: `InlineJobExecutor`, `%pdb on`, common pitfalls, ...


API Reference
-------------

.. automodule:: libertem.udf.base
   :members:
   :special-members: __init__
   :exclude-members: UDFTask,UDFRunner
