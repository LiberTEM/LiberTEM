User-defined functions: advanced topics
=======================================

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


If you look closely, you may notice that for each frame, all elements from :code:`weights` are accessed.
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

For example, in case of K2IS data, frames have a shape of :code:`(1860, 2048)`. When reading them
with the tiled strategy, a single tile will contain data from 16 subsequent frames, and each
rectangle has a shape of :code:`(930, 16)` (which happens to be the natural block size for K2IS data).
So the tiles will have a shape of :code:`(16, 930, 16)`, and processing 16 frames from the data set
means reading 256 individual tiles.

Loading a tile of this size as float32 data
still fits comfortably into usual L3 CPU caches (~1MB), and thus enables efficient processing.
As a comparison, a whole :code:`(1860, 2048)` frame is about 15MB large, and accessing it repeatedly
means having to load data from the slower main memory.

Note: you may have noticed that we talk about block sizes of 1MB as efficient in the L3 cache,
but many CPUs have larger L3 caches. As the L3 cache is shared between cores, and LiberTEM tries
to use multiple cores, the effectively available L3 cache has to be divided by number of cores.

Real-world example
~~~~~~~~~~~~~~~~~~

The :class:`~libertem.udf.blobfinder.SparseCorrelationUDF` uses :meth:`~libertem.udf.UDFTileMixin.process_tile` to implement a custom version of a :class:`~libertem.job.masks.ApplyMasksJob` that works on log-scaled data. The mask stack is stored in a :class:`libertem.job.mask.MaskContainer` as part of the task data. Note how the :class:`~libertem.common.Slice` :code:`tile_slice` argument is used to extract the region from the mask stack that matches the tile using the facilities of a :class:`~libertem.job.masks.MaskContainer`. After reshaping, transposing and log scaling the tile data into the right memory layout, the mask stack is applied to the data with a dot product. The result is *added* to the buffer in order to merge it with the results of the other tiles because addition is the correct merge function for a dot product. Other operations would require a different merge function here, for example :meth:`numpy.max()` if a global maximum is to be calculated.

.. code-block:: python

    def process_tile(self, tile, tile_slice):
        c = self.task_data['mask_container']
        tile_t = np.zeros(
            (np.prod(tile.shape[1:]), tile.shape[0]),
            dtype=tile.dtype
        )
        log_scale(tile.reshape((tile.shape[0], -1)).T, out=tile_t)

        sl = c.get(key=tile_slice, transpose=False)
        self.results.corr[:] += sl.dot(tile_t).T

Post-processing
~~~~~~~~~~~~~~~

Post-processing allows to perform additional processing steps once the data of a partition is completely processed with :meth:`~libertem.udf.UDFFrameMixin.process_frame`, :meth:`~libertem.udf.UDFTileMixin.process_tile` or :meth:`~libertem.udf.UDFPartitionMixin.process_partition`. Post-processing is particularly relevant for tiled processing since that allows to combine the performance benefits of tiled processing for a first reduction step with subsequent steps that require reduced data from complete frames or even a complete partition.

Real-world example from :class:`~libertem.udf.blobfinder.SparseCorrelationUDF` which evaluates the correlation maps that have been generated with the dot product in the previous processing step and places the results in additional result buffers:

.. code-block:: python

    def postprocess(self):
        steps = 2 * self.params.steps + 1
        corrmaps = self.results.corr.reshape((
            -1,  # frames
            len(self.params.peaks),  # peaks
            steps,  # Y steps
            steps,  # X steps
        ))
        peaks = self.params.peaks
        r = self.results
        for f in range(corrmaps.shape[0]):
            for p in range(len(self.params.peaks)):
                corr = corrmaps[f, p]
                center, refined, peak_value, peak_elevation = evaluate_correlation(corr)
                abs_center = _shift(center, peaks[p], self.params.steps).astype('u2')
                abs_refined = _shift(refined, peaks[p], self.params.steps).astype('float32')
                r.centers[f, p] = abs_center
                r.refineds[f, p] = abs_refined
                r.peak_values[f, p] = peak_value
                r.peak_elevations[f, p] = peak_elevation


Partition processing
--------------------

Some algorithms can benefit from processing entire partitions, for example if they require several passes over the data. In most cases, :ref:`tiled processing<tiled>` will be faster because it uses the L3 cache more efficiently. For that reason, per-partition processing should only be used if there are clear indications for it. Implementing :meth:`~libertem.udf.UDFPartitionMixin.process_partition` activates per-partition processing for an UDF.

Precedence
----------

The UDF interface looks for methods in the order :meth:`~libertem.udf.UDFTileMixin.process_tile`, :meth:`~libertem.udf.UDFFrameMixin.process_frame`, :meth:`~libertem.udf.UDFPartitionMixin.process_partition`. For now, the first in that order is executed. In the future, composition of UDFs may allow to use different methods depending on the circumstances. :meth:`~libertem.udf.UDFTileMixin.process_tile` is the most general method and allows by-frame and by-partition processing as well.

