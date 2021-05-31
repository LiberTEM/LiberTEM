.. _`advanced udf`:

User-defined functions: advanced topics
---------------------------------------

.. testsetup:: *

    import numpy as np
    from libertem import api
    from libertem.executor.inline import InlineJobExecutor
    from libertem.viz.base import Dummy2DPlot

    ctx = api.Context(executor=InlineJobExecutor(), plot_class=Dummy2DPlot)
    data = np.random.random((16, 16, 32, 32)).astype(np.float32)
    dataset = ctx.load("memory", data=data, sig_dims=2)
    roi = np.random.choice([True, False], dataset.shape.nav)

The UDF interface offers a wide range of features to help implement advanced
functionality and to optimize the performance of an UDF. These features are
optional in order to keep UDFs that don't need them simple.

See :ref:`user-defined functions` for an introduction to basic topics.

.. _tiled:

Tiled processing
----------------

Many operations can be significantly optimized by working on stacks of frames.
You can often perform `loop nest optimization
<https://en.wikipedia.org/wiki/Loop_nest_optimization>`_ to improve the
`locality of reference <https://en.wikipedia.org/wiki/Locality_of_reference>`_,
for example using `numba <https://numba.pydata.org/>`_, or using an optimized
NumPy function.

As an example, applying a gain map and subtracting dark frames can be up to an
order of magnitude faster when properly optimized compared to a naive NumPy
implementation. These optimizations are only possible if you have access to data
from more than one frame.

For very large frames, another problem arises: a stack of frames would be too
large to efficiently handle, as it would no longer fit into even the L3 cache,
which is the largest cache in most CPUs. For these cases, we support a tiled
reading and processing strategy. Tiled means we slice the frame into disjoint
rectangular regions. A tile then is the data from a single rectangular region
for multiple frames.

For example, in case of K2IS data, frames have a shape of :code:`(1860, 2048)`.
When reading them with the tiled strategy, a single tile will contain data from
16 subsequent frames, and each rectangle has a shape of :code:`(930, 16)`, which
is the natural block size for K2IS data. That means the tiles will have a shape
of :code:`(16, 930, 16)`, and processing 16 frames from the data set means
reading 256 individual tiles.

Loading a tile of this size as float32 data still fits comfortably into usual L3
CPU caches (~1MB per core), and thus enables efficient processing. As a comparison, a
whole :code:`(1860, 2048)` frame is about 15MB large, and accessing it
repeatedly means having to load data from the slower main memory.

.. note::
    You may have noticed that we talk about block sizes of 1MB as efficient in
    the L3 cache, but many CPUs have larger L3 caches. As the L3 cache is shared
    between cores, and LiberTEM tries to use multiple cores, the effectively
    available L3 cache has to be divided by number of cores.

.. _`slice example`:

Real-world example
~~~~~~~~~~~~~~~~~~

The :class:`libertem_blobfinder.udf.correlation.SparseCorrelationUDF` uses
:meth:`~libertem.udf.base.UDFTileMixin.process_tile` to implement a custom version of
a :class:`~libertem.udf.masks.ApplyMasksUDF` that works on log-scaled data. The
mask stack is stored in a :class:`libertem.common.container.MaskContainer` as part of
the task data. Note how the :code:`self.meta.slice` property of type
:class:`~libertem.common.slice.Slice` is used to extract the region from the mask
stack that matches the tile using the facilities of a
:class:`~libertem.common.container.MaskContainer`. After reshaping, transposing and log
scaling the tile data into the right memory layout, the mask stack is applied to
the data with a dot product. The result is *added* to the buffer in order to
merge it with the results of the other tiles because addition is the correct
merge function for a dot product. Other operations would require a different
merge function here, for example :meth:`numpy.max()` if a global maximum is to
be calculated.

.. testsetup::

    from libertem_blobfinder.base.correlation import log_scale

.. testcode::

    def process_tile(self, tile):
        tile_slice = self.meta.slice
        c = self.task_data['mask_container']
        tile_t = np.zeros(
            (np.prod(tile.shape[1:]), tile.shape[0]),
            dtype=tile.dtype
        )
        log_scale(tile.reshape((tile.shape[0], -1)).T, out=tile_t)

        sl = c.get(key=tile_slice, transpose=False)
        self.results.corr[:] += sl.dot(tile_t).T

.. _`udf post processing`:

Partition processing
--------------------

Some algorithms can benefit from processing entire partitions, for example if
they require several passes over the data. In most cases, :ref:`tiled
processing<tiled>` will be faster because it uses the L3 cache more efficiently.
For that reason, per-partition processing should only be used if there are clear
indications for it. Implementing
:meth:`~libertem.udf.base.UDFPartitionMixin.process_partition` activates
per-partition processing for an UDF.

Precedence
----------

The UDF interface looks for methods in the order
:meth:`~libertem.udf.base.UDFTileMixin.process_tile`,
:meth:`~libertem.udf.base.UDFFrameMixin.process_frame`,
:meth:`~libertem.udf.base.UDFPartitionMixin.process_partition`. For now, the first in
that order is executed. In the future, composition of UDFs may allow to use
different methods depending on the circumstances.
:meth:`~libertem.udf.base.UDFTileMixin.process_tile` is the most general method and
allows by-frame and by-partition processing as well.

Post-processing of partition results
------------------------------------

Post-processing allows to perform additional processing steps once the data of a
partition is completely processed with
:meth:`~libertem.udf.base.UDFFrameMixin.process_frame`,
:meth:`~libertem.udf.base.UDFTileMixin.process_tile` or
:meth:`~libertem.udf.base.UDFPartitionMixin.process_partition`. Post-processing is
particularly relevant for tiled processing since that allows to combine the
performance benefits of tiled processing for a first reduction step with
subsequent steps that require reduced data from complete frames or even a
complete partition.

Real-world example from
:class:`libertem_blobfinder.udf.correlation.SparseCorrelationUDF` which
evaluates the correlation maps that have been generated with the dot product in
the previous processing step and places the results in additional result
buffers:

.. testsetup::

    from libertem_blobfinder.base.correlation import evaluate_correlations

.. testcode::

    def postprocess(self):
        steps = 2 * self.params.steps + 1
        corrmaps = self.results.corr.reshape((
            -1,  # frames
            len(self.params.peaks),  # peaks
            steps,  # Y steps
            steps,  # X steps
        ))
        peaks = self.params.peaks
        (centers, refineds, peak_values, peak_elevations) = self.output_buffers()
        for f in range(corrmaps.shape[0]):
            evaluate_correlations(
                corrs=corrmaps[f], peaks=peaks, crop_size=self.params.steps,
                out_centers=centers[f], out_refineds=refineds[f],
                out_heights=peak_values[f], out_elevations=peak_elevations[f]
            )

The :meth:`libertem.udf.base.UDFPreprocessMixin.postprocess` method is called
for each partition on the worker process, before the results from different
partitions have been merged.

.. _`udf final post processing`:

Post-processing after merging
-----------------------------

If you want to implement a post-processing step that is run on the main node
after merging result buffers, you can override
:meth:`libertem.udf.base.UDF.get_results`:

.. testsetup::

    from libertem.udf import UDF

.. testcode::

    class AverageUDF(UDF):
        """
        Like SumUDF, but also computes the average
        """
        def get_result_buffers(self):
            return {
                'sum': self.buffer(kind='sig', dtype=np.float32),
                'num_frames': self.buffer(kind='single', dtype=np.uint64),
                'average': self.buffer(kind='sig', dtype=np.float32, use='result_only'),
            }

        def process_frame(self, frame):
            self.results.sum[:] += frame
            self.results.num_frames[:] += 1

        def merge(self, dest, src):
            dest.sum[:] += src.sum
            dest.num_frames[:] += src.num_frames

        def get_results(self):
            return {
                # NOTE: 'sum' omitted here, will be returned unchanged
                'average': self.results.sum / self.results.num_frames,
            }

    ctx.run_udf(dataset=dataset, udf=AverageUDF())

Note that :meth:`UDF.get_result_buffers` returns a placeholder entry for the
:code:`average` result using :code:`use='result_only'`, which is then filled in
:code:`get_results`.  We don't need to repeat those buffers that should be
returned unchanged; if you want to omit a buffer from the results completely,
you can declare it as private with :code:`self.buffer(..., use='private')` in
:code:`get_result_buffers`.

:meth:`UDF.get_results` should return the results as a dictionary of numpy
arrays, with the keys matching those returned by
:meth:`UDF.get_result_buffers`.

When returned from :meth:`Context.run_udf`, all results are wrapped into
:code:`BufferWrapper` instances. This is done primarily to get convenient
access to a version of the result that is suitable for visualization, even if
a :code:`roi` was used, but still allow access to the raw result using
:attr:`BufferWrapper.raw_data` attribute.

The detailed rules for buffer declarations, :code:`get_result_buffers` and :code:`get_results` are:

1) All buffers are declared in :code:`get_result_buffers`
2) If a buffer is only computed in :code:`get_results`, it should be marked via
   :code:`use='result_only'` so it isn't allocated on workers
3) If a buffer is only used as intermediary result, it should be marked via :code:`use='private'`
4) Not including a buffer in :code:`get_results` means it will either be passed on
   unchanged, or dropped if :code:`use='private'`
5) It's an error to omit an :code:`use='result_only'` buffer in :code:`get_results`
6) It's an error to include a :code:`use='private'` buffer in :code:`get_results`
7) All results are returned from :code:`Context.run_udf` as :code:`BufferWrapper` instances
8) By default, if :code:`get_results` is not implemented, :code:`use='private'` buffers are dropped,
   and others are passed through unchanged

.. versionadded:: 0.7.0
   :meth:`UDF.get_results` and the :code:`use` argument for :meth:`UDF.buffer` were added.

Pre-processing
---------------

Pre-processing allows to initialize result buffers before processing or merging.
This is particularly useful to set up :code:`dtype=object` buffers, for example
ragged arrays, or to initialize buffers for operations where the neutral element
is not 0. :meth:`libertem.udf.base.UDFPreprocessMixin.preprocess` is executed after
all buffers are allocated, but before the data is processed. On the worker nodes
it is executed with views set for the whole partition masked by the current ROI.
On the central node it is executed with views set for the whole dataset masked
by the ROI. 

.. versionadded:: 0.3.0

.. versionchanged:: 0.5.0
    :meth:`libertem.udf.base.UDFPreprocessMixin.preprocess` is executed on the main
    node, too. Views for aux data are set correctly on the main node. Previously,
    it was only executed on the worker nodes.

AUX data
--------

If a parameter is an instance of :class:`~libertem.common.buffers.BufferWrapper`
that was created using the :meth:`~libertem.udf.base.UDF.aux_data` class method, the
UDF interface will interpret it as auxiliary data. It will set the views for
each tile/frame/partition accordingly so that accessing the parameter returns a
view of the auxiliary data matching the data portion that is currently being
processed. That way, it is possible to pass parameters individually for each
frame or to mask the signal dimension.

Note that the :class:`~libertem.common.buffers.BufferWrapper` instance for AUX
data should always be created using the :meth:`~libertem.udf.base.UDF.aux_data` class
method and not directly by instantiating a
:class:`~libertem.common.buffers.BufferWrapper` since
:meth:`~libertem.udf.base.UDF.aux_data` ensures that it is set up correctly.

For masks in the signal dimension that are used for dot products in combination
with per-tile processing, a :class:`~libertem.common.container.MaskContainer` allows
to use more advanced slicing and transformation methods targeted at preparing
mask stacks for optimal dot product performance.

Task data
---------

A UDF can generate task-specific intermediate data on the worker nodes by
defining a :meth:`~libertem.udf.base.UDF.get_task_data` method. The result is
available as an instance of :class:`~libertem.udf.base.UDFData` in
:code:`self.task_data`. Depending on the circumstances, this can be more
efficient than making the data available as a parameter since it avoids
pickling, network transport and unpickling.

This non-trivial example from
:class:`libertem_blobfinder.udf.correlation.SparseCorrelationUDF` creates
a :class:`~libertem.common.container.MaskContainer` based on the parameters in
:code:`self.params`. This :class:`~libertem.common.container.MaskContainer` is then
available as :code:`self.task_data['mask_container']` within the processing
functions.

.. testsetup::

    from libertem.common.container import MaskContainer
    import libertem.masks as masks

.. testcode::

    def get_task_data(self):
        match_pattern = self.params.match_pattern
        crop_size = match_pattern.get_crop_size()
        size = (2 * crop_size + 1, 2 * crop_size + 1)
        template = match_pattern.get_mask(sig_shape=size)
        steps = self.params.steps
        peak_offsetY, peak_offsetX = np.mgrid[-steps:steps + 1, -steps:steps + 1]

        offsetY = self.params.peaks[:, 0, np.newaxis, np.newaxis] + peak_offsetY - crop_size
        offsetX = self.params.peaks[:, 1, np.newaxis, np.newaxis] + peak_offsetX - crop_size

        offsetY = offsetY.flatten()
        offsetX = offsetX.flatten()

        stack = functools.partial(
            masks.sparse_template_multi_stack,
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

.. testcleanup::

    from libertem_blobfinder.udf.correlation import SparseCorrelationUDF
    from libertem_blobfinder.common.patterns import RadialGradient

    class TestUDF(SparseCorrelationUDF):
        pass

    # Override methods with functions that are defined above

    TestUDF.process_tile = process_tile
    TestUDF.postprocess = postprocess
    TestUDF.get_task_data = get_task_data

    u = TestUDF(
        peaks=np.array([(8, 8)]),
        match_pattern=RadialGradient(2),
        steps=3
    )
    ctx.run_udf(dataset=dataset, udf=u)

Meta information
----------------

Advanced processing routines may require context information about the processed
data set, ROI and current data portion being processed. This information is
available as properties of the :attr:`libertem.udf.base.UDF.meta` attribute of type
:class:`~libertem.udf.base.UDFMeta`.

Input data shapes and types
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Common applications include allocating buffers with a :code:`dtype` or shape
that matches the dataset or partition via
:attr:`~libertem.udf.base.UDFMeta.dataset_dtype`,
:attr:`~libertem.udf.base.UDFMeta.input_dtype`,
:attr:`~libertem.udf.base.UDFMeta.dataset_shape` and
:attr:`~libertem.udf.base.UDFMeta.partition_shape`.

Device class
~~~~~~~~~~~~

The currently used compute device class can be accessed through
:attr:`libertem.udf.base.UDFMeta.device_class`. It defaults to 'cpu' and can be 'cuda'
for UDFs that make use of :ref:`udf cuda` support.

ROI and current slice
~~~~~~~~~~~~~~~~~~~~~

For more advanced applications, the ROI and currently processed data portion are
available as :attr:`libertem.udf.base.UDFMeta.roi` and
:attr:`libertem.udf.base.UDFMeta.slice`. This allows to replace the built-in masking
behavior of :class:`~libertem.common.buffers.BufferWrapper` for result buffers
and aux data with a custom implementation. The :ref:`mask container for tiled
processing example<slice example>` makes use of these attributes to employ a
:class:`libertem.common.container.MaskContainer` instead of a :code:`shape="sig"`
buffer in order to optimize dot product performance and support sparse masks.

The slice is in the reference frame of the dataset, masked by the current ROI,
with flattened navigation dimension. This example illustrates the behavior by
implementing a custom version of the :ref:`simple "sum over sig" example
<sumsig>`. It allocates a custom result buffer that matches the navigation
dimension as it appears in processing:

.. testcode::

    import numpy as np

    from libertem.udf import UDF

    class PixelsumUDF(UDF):
        def get_result_buffers(self):
            if self.meta.roi is not None:
                navsize = np.count_nonzero(self.meta.roi)
            else:
                navsize = np.prod(self.meta.dataset_shape.nav)
            return {
                'pixelsum_nav_raw': self.buffer(
                    kind="single",
                    dtype=self.meta.dataset_dtype,
                    extra_shape=(navsize, ),
                )
            }

        def merge(self, dest, src):
            dest.pixelsum_nav_raw[:] += src.pixelsum_nav_raw

        def process_frame(self, frame):
            np_slice = self.meta.slice.get(nav_only=True)
            self.results.pixelsum_nav_raw[np_slice] = np.sum(frame)

.. testcleanup::

    pixelsum = PixelsumUDF()
    res = ctx.run_udf(dataset=dataset, udf=pixelsum, roi=roi)

    assert np.allclose(res['pixelsum_nav_raw'].data, dataset.data[roi].sum(axis=(1, 2)))

Coordinates
~~~~~~~~~~~

The coordinates of the current frame, tile or partition within the true dataset
navigation dimension, as opposed to the current slice that is given in flattened
nav dimensions with applied ROI, is available through
:attr:`~libertem.udf.base.UDFMeta.coordinates`. The following UDF simply
collects the coordinate info for demonstration purposes. A real-world example
that uses the coordinates is `the UDF implementation of single side band
ptychography
<https://github.com/Ptychography-4-0/ptychography/blob/master/src/ptychography40/reconstruction/ssb/udf.py>`_.

.. testcode::

    import numpy as np

    from libertem.udf import UDF

    class CoordUDF(UDF):
        def get_result_buffers(self):
            # Declare a buffer that fits the coordinates,
            # i.e. one int per nav axis for each nav position
            nav_dims = len(self.meta.dataset_shape.nav)
            return {
                'coords': self.buffer(
                    kind="nav",
                    dtype=int,
                    extra_shape=(nav_dims, ),
                )
            }

        def process_tile(self, tile):
            # Simply copy the coordinates into
            # the result buffer
            self.results.coords[:] = self.meta.coordinates

    my_roi = np.zeros(dataset.shape.nav, dtype=bool)
    my_roi[7, 13] = True
    my_roi[11, 3] = True

    res = ctx.run_udf(
        dataset=dataset,
        udf=CoordUDF(),
        roi=my_roi
    )

    assert np.all(
        res['coords'].raw_data == np.array([(7, 13), (11, 3)])
    )

.. _`udf dtype`:

Preferred input dtype
---------------------

UDFs can override :meth:`~libertem.udf.base.UDF.get_preferred_input_dtype` to
indicate a "lowest common denominator" compatible dtype. The actual input dtype
is determined by combining the indicated preferred dtype with the input
dataset's native dtype using :func:`numpy.result_type`. The default preferred
dtype is :attr:`numpy.float32`. Returning :attr:`UDF.USE_NATIVE_DTYPE`, which is
currently identical to :code:`numpy.bool`, will switch to the dataset's native
dtype since :code:`numpy.bool` behaves as a neutral element in
:func:`numpy.result_type`.

If an UDF requires a specific dtype rather than only preferring it, it should
override this method and additionally check the actual input type, throw an
error when used incorrectly and/or implement a meaningful conversion in its
processing routine since indicating a preferred dtype doesn't enforce it. That
way, unsafe conversions are performed explicitly in the UDF rather than
indirectly in the back-end.

.. versionadded:: 0.4.0

.. _`udf cuda`:

CuPy support
------------

LiberTEM can use CUDA devices through `CuPy <https://cupy.chainer.org/>`_. Since
CuPy largely replicates the NumPy array interface, any UDF that uses NumPy for
its main processing can likely be ported to use both CPUs and CUDA devices in
parallel. Some adjustments are often necessary to account for minor differences
between NumPy and CuPy. CuPy is most beneficial for compute-heavy tasks with
good CUDA math library support such as large Fourier transforms or matrix
products.

In order to activate CuPy processing, a UDF can overwrite the
:meth:`~libertem.udf.base.UDF.get_backends` method. By default this returns
:code:`('numpy',)`, indicating only NumPy support. By returning :code:`('numpy',
'cupy')` or :code:`('cupy',)`, a UDF activates being run on both CUDA and CPU
workers, or exclusively on CUDA workers. Using :code:`cuda` instead of
:code:`cupy` schedules on CUDA workers, but without using the CuPy library. This
is useful for running code that uses CUDA in a different way, for example
integration of C++ CUDA code, and allows to skip installation of CuPy in this
situation.

The :attr:`libertem.udf.base.UDF.xp` property points to the :code:`numpy` or
:code:`cupy` module, depending which back-end is currently used. By using
:code:`self.xp` instead of the usual :code:`np` for NumPy, one can write UDFs
that use the same code for CUDA and CPU processing.

Result buffers can be declared as device arrays by setting
:code:`self.buffer(..., where='device')` in
:meth:`~libertem.udf.base.UDF.get_result_buffers`. That allows to keep data in
the device until a partition is completely processed and the result is exported
to the leader node.

The input argument for :code:`process_*()` functions is already provided as a
CuPy array instead of NumPy array if CuPy is used.

A UDF should only use one GPU at a time. If :code:`cupy` is used, the correct
device to use is set within CuPy in the back-end and should not be modified in
the UDF itself. If :code:`cuda` is used, it is the responsibility of the user to
set the device ID to the value returned by
:meth:`libertem.common.backend.get_use_cuda`. The environment variable
:code:`CUDA_VISIBLE_DEVICES` can be set `before` any CUDA library is loaded to
control which devices are visible.

The :meth:`~libertem.api.Context.run_udf` method allows setting the
:code:`backends` attribute to :code:`('numpy',)` :code:`('cupy',)` or :code:`('cuda',)` to
restrict execution to CPU-only or CUDA-only on a hybrid cluster. This is mostly
useful for testing.

.. versionadded:: 0.6.0

.. _auto UDF:

Auto UDF
--------

The :class:`~libertem.udf.AutoUDF` class and :meth:`~libertem.api.Context.map`
method allow to run simple functions that accept a frame as the only parameter
with an auto-generated :code:`kind="nav"` result buffer over a dataset ad-hoc
without defining an UDF class. For more advanced processing, such as custom
merge functions, post-processing or performance optimization through tiled
processing, defining an UDF class is required.

As an alternative to Auto UDF, you can use the
:meth:`~libertem.contrib.daskadapter.make_dask_array` method to create
a `dask.array <https://docs.dask.org/en/latest/array.html>`_ from
a :class:`~libertem.io.dataset.base.DataSet` to perform calculations. See
:ref:`Integration with Dask arrays<daskarray>` for more details.

The :class:`~libertem.udf.AutoUDF` class determines the output shape and type
by calling the function with a mock-up frame of the same type and shape as
a real detector frame and converting the return value to a NumPy array. The
:code:`extra_shape` and :code:`dtype` parameters for the result buffer are
derived automatically from this NumPy array.

Additional constant parameters can be passed to the function via
:meth:`functools.partial`, for example. The return value should be much smaller
than the input size for this to work efficiently.

Example: Calculate sum over the last signal axis.

.. testcode::

    import functools

    result = ctx.map(
        dataset=dataset,
        f=functools.partial(np.sum, axis=-1)
    )

    # or alternatively:
    from libertem.udf import AutoUDF

    udf = AutoUDF(f=functools.partial(np.sum, axis=-1))
    result = ctx.run_udf(dataset=dataset, udf=udf)
