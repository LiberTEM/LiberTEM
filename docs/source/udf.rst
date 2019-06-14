User-defined functions
======================

A common case for analysing big EM data sets is running a reduction operation
on each frame of the data set. The user-defined functions (UDF) interface
of LiberTEM allows users to run their own reduction functions easily, without having
to worry about parallelizing, I/O, the details of buffer management and so on. 


By-frame processing
-------------------
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
