How does I/O work in LiberTEM?
==============================

Many algorithms benefit from :ref:`tiled` where the same slice of the signal
dimension is processed for several frames in a row. In many cases, algorithms
have specific minimum and maximum sizes in signal dimension, navigation
dimension or total size where they operate efficiently. Smaller sizes might
increase overheads, while larger sizes might reduce cache efficiency.

At the same time, file formats might operate well within specific size and
shape limits. The :ref:`k2is` raw format is a prime example where data is saved
in tiled form and can be processed efficiently in specific tile sizes and
shapes that follow the native layout. Furthermore, some formats require
decoding or corrections by the CPU, such as :ref:`frms6`, where tiles that fit
the L3 cache can speed up subsequent processing steps. Requirements from the
I/O method such as alignment and efficient block sizes are taken into account
as well.

The LiberTEM I/O back-end negotiates a tiling scheme between UDF and dataset
that fulfills requirements from both UDF and dataset side as far as possible.
However, it is not always guaranteed that the supplied data will fall within
the requested limits.

.. versionadded:: 0.6.0
  This guide is written for version 0.6.0

High-level overview
~~~~~~~~~~~~~~~~~~~

- Data is split into partitions which are read from independently. Usually
  they split the navigation axis.
- For each partition, the :code:`UDFRunner` negotiates a :code:`TilingScheme` using the
  :code:`Negotiator` class
- The :code:`TilingScheme` is then passed on to :code:`Partition.get_tiles`,
  which then yields :code:`DataTiles` that match the given
  :code:`TilingScheme`.
- Under the hood, the :code:`Partition`...
   - instantiates an :code:`IOBackend`, which has a reference to a :code:`Decoder`
   - generates read ranges, which are passed on to the :code:`IOBackend`
   - delegates :code:`get_tiles` to the :code:`IOBackend`
- The I/O process can be influenced by passing a subclass
  of :code:`FileSet` to the :code:`Partition` and overriding :code:`FileSet.get_read_ranges`,
  implementing a :code:`Decoder`, or even completely overriding
  the :code:`Partition.get_tiles` functionality.
- There are currently two I/O backends implemented: :code:`MMapBackend` and :code:`BufferedBackend`,
  which are useful for different storage media.
- :code:`MMapBackend.get_tiles` has two modes of operation: either it returns a reference to the
  tiles "straight" from the file, without copying or decoding, or it
  uses the read ranges and copies/decodes the tiles in smaller units.
- When reading the tiles "straight", the read ranges are not used, instead
  only the slice information for each tile is used. That also means that this
  mode only works for very simple formats, when reading without a :code:`roi`
  and when not doing any :code:`dtype` conversion or decoding.
- For most formats, a :code:`sync_offset` can be specified, which can be used to
  correct for synchronization errors by inserting blank frames,
  or ignoring one or more frames, at the beginning or at the end of the data set.

Read ranges
-----------

In :code:`FileSet.get_read_ranges`, the reading parameters (:code:`TilingScheme`, :code:`roi` etc.)
are translated into one or more byte ranges (offset, length) for each tile.
You can imagine it as translating pixel/element positions into byte offsets.

Each range corresponds to a read operation on a single file, and with multiple
read ranges per tile, ranges for a single tile can correspond to reads from multiple files.
This is important when reading from a data set with many small files - we can
still generate deep tiles for efficient processing.

There are some built-in common parameters in :code:`FileSet`, like
:code:`frame_header_bytes`, :code:`frame_footer_bytes`, which can be used to easily
implement formats where the reading just needs to skip a few bytes for each
frame header/footer.

If you need more influence over how data is read, you can override
:code:`FileSet.get_read_ranges` and return your own read ranges. You can use
the :code:`make_get_read_ranges` function to re-use much of the tiling logic,
or implement this yourself. Using :code:`make_get_read_ranges` you can either
override just the :code:`px_to_bytes` part, or :code:`read_ranges_tile_block` for whole
tile blocks. This is done by passing in njit-ed functions to :code:`make_get_read_ranges`.
:code:`make_get_read_ranges` should only be called on module-level to enable
caching of the numba compilation.

Read ranges are generated as an array with the following shape::

    :code:`(number_of_tiles, rr_per_tile, rr_num_entries)`

:code:`rr_per_tile` here is the maximum number of read ranges per tile - there
can be tiles that are smaller than this, for example at the end of a partition.
:code:`rr_num_entries` is at least 3 and contains at least the values
:code:`(file_idx, start, stop)`. This means to read :code:`stop - start`
bytes, beginning at offset :code:`start`, from the file :code:`file_idx`
in the corresponding :code:`FileSet`.

Overriding :code:`DataSet`\ s are free to add additional fields to the end, for
example if the decoding functions need additional information.

As an example when you would generate custom read ranges, have a look at the
implementations for MIB, K2IS, and FRMS6 - they may not have a direct 1:1 mapping
to a numpy :code:`dtype`, or the pixels may need to be re-ordered after decoding.

Notes for implementing a :class:`~libertem.io.dataset.base.DataSet`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Read file header(s) in :meth:`~libertem.io.dataset.base.DataSet.initialize` -
  make sure to do the actual I/O in a function dispatched via the
  :code:`JobExecutor` that is passed to :code:`initialize`.
  See also :ref:`os mismatch` regarding platform-dependent code.
- Implement :meth:`~libertem.io.dataset.base.DataSet.check_valid` - this will
  be run on a worker node
- Implement :meth:`~libertem.io.dataset.base.DataSet.get_msg_converter` - the
  :class:`~libertem.web.messages.MessageConverter` class returned is responsible
  for parsing parameters passed to the Web API and converting them to a Python
  representation that can be passed to the
  :class:`~libertem.io.dataset.base.DataSet` constructor.
- Implement :meth:`~libertem.io.dataset.base.DataSet.get_cache_key` - the cache
  key must be different for :code:`DataSet`\ s that return different data.
- Implement :meth:`~libertem.io.dataset.base.DataSet.get_partitions`. You may
  want to use the helper function
  :meth:`~libertem.io.dataset.base.DataSet.get_slices` to generate
  slices for a specified number of partitions.
  :meth:`~libertem.io.dataset.base.DataSet.get_partitions` should yield either
  :class:`~libertem.io.dataset.base.BasePartition` instances or instances of
  your own subclass (see below). The same is true for the
  :class:`~libertem.io.dataset.base.FileSet` that is passed to each
  partition - you possibly have to implement your own subclass.

Subclass :class:`~libertem.io.dataset.base.BasePartition`
---------------------------------------------------------

- Implement :meth:`~libertem.io.dataset.base.BasePartition._get_decoder` to return
  an instance of :class:`~libertem.io.dataset.base.Decoder`. Only needed if
  the data is saved in a data type that is not directly understood by numpy
  or numba. See below for details.
- Implement :meth:`~libertem.io.dataset.base.BasePartition.get_base_shape`. This
  is only needed if the data format imposes any constraints on how the data can be
  read in an efficient manner, for example if data is saved in blocks. The tileshape
  that is negotiated before reading will be a multiple of the base shape in
  all dimensions.
- Implement :meth:`~libertem.io.dataset.base.BasePartition.adjust_tileshape`. This
  is needed if you need to "veto" the generated tileshape, for example if your dataset
  has constraints that can't be expressed by the base shape.
- Override :meth:`~libertem.io.dataset.base.BasePartition.get_tiles` if you need to
  use completely custom I/O logic.

Implementing a :class:`~libertem.io.dataset.base.Decoder`
---------------------------------------------------------

This may be needed if the raw data is not directly supported
by numpy or numba. Mostly your decoder will return a different
:code:`decode` function in :meth:`~libertem.io.dataset.base.Decoder.get_decode`.
You can also return different decode functions, depending on the
concrete data set you are currently reading. For example, this may be needed if there
are different data representations generated by different detector modes.
You can also instruct the :code:`IOBackend` to clear the read
buffer before calling :code:`decode` by returning :code:`True` from 
:meth:`~libertem.io.dataset.base.Decoder.do_clear`. This can be needed
if different read ranges contribute to the same part of the output buffer
and the :code:`decode` function accumulates into the buffer instead of slice-assigning.

The :code:`decode` function will be called for each read range that was
generated by the :code:`get_read_ranges` method described above.