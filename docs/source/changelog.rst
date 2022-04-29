Changelog
=========

.. testsetup:: *

    from libertem import api
    from libertem.executor.inline import InlineJobExecutor

    ctx = api.Context(executor=InlineJobExecutor())
    dataset = ctx.load("memory", datashape=(16, 16, 16, 16), sig_dims=2)

.. _continuous:

0.10.0.dev0
###########

.. toctree::
  :glob:

  changelog/*/*

.. _`v0-9-2`:
.. _latest:

0.9.2 / 2022-04-28
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6502282.svg
   :target: https://doi.org/10.5281/zenodo.6502282

This is a bugfix release with two small fixes:

* Example notebook: compatibility with HyperSpy 1.7.0 :pr:`1242`
* Compatibility of CoM auto button with Jupyter server proxy :pr:`1220`

.. _`v0-9-0`:

0.9.0 / 2022-02-17
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.6125247.svg
   :target: https://doi.org/10.5281/zenodo.6125247

We are most happy to announce **full Dask array integration** with this release!
Many thanks to Matthew Bryan who implemented major parts of this non-trivial
feature. Most notably, HyperSpy lazy signals and LiberTEM can now be combined
seamlessly. See :ref:`dask` for details and an example!

This enables the following applications:

* Use HyperSpy file readers and other readers that create Dask arrays for
  LiberTEM.
* Create an ad-hoc file reader for LiberTEM by just building a Dask array. This
  is often simpler than implementing a native LiberTEM dataset, at the expense
  of performance.
* Use LiberTEM file readers for HyperSpy and other software that works with
  Dask arrays.
* Use the same implementation of an algorithm for live processing with LiberTEM,
  offline processing with LiberTEM, and offline processing with HyperSpy.
* Simplify implementation of complex processing routines on Dask arrays. That
  includes, for example, routines that are not purely implemented with NumPy
  array operations and produce complex output or are not compatible with all
  Dask array chunking schemes. Here, LiberTEM UDFs offer a more powerful and
  versatile interface than Dask's native `map_blocks()
  <https://docs.dask.org/en/latest/generated/dask.array.map_blocks.html>`_
  interface.
* Chain processing steps together using Dask arrays for intermediate results,
  including using the output of one UDF as input for another UDF. Dask arrays
  allow working with large intermediate results efficiently since they can
  remain on the workers.

Specifically, the Dask integration encompasses the following features:

* Create LiberTEM datasets from Dask arrays via the :ref:`daskds` (:pr:`1137`).
* Create Dask arrays from LiberTEM UDF results using the
  :class:`~libertem.executor.delayed.DelayedJobExecutor`. A UDF can define a
  :meth:`~libertem.udf.base.UDFMergeAllMixin.merge_all` method in addition to
  the usual :meth:`~libertem.udf.base.UDF.merge` to improve performance. See
  :ref:`merge_all` for details (:pr:`1170`)!
* Create Dask arrays directly from LiberTEM datasets using
  :func:`libertem.contrib.daskadapter.make_dask_array`, which is already
  possible since release 0.2.
* Executor options to improve integration, see :ref:`scheduler` and
  :ref:`executors` (:pr:`1170`, :issue:`1146,922`).

Please note that these features are still experimental and cover a large space
of possible uses and parameters. Expect the unexpected! Tests, feedback and
improvements are highly appreciated.

Other changes in this release:

New features
------------

* Experimental helper function :meth:`libertem.analysis.com.guess_corrections`
  to guess parameters for Center of Mass analysis (:pr:`1111`).
* GUI interface for the COM analysis to call :meth:`libertem.analysis.com.guess_corrections`
  and update the GUI parameters from the result (:pr:`1172`).
* Support for some MIB Quad formats. All integer formats should be supported and
  were tested with :code:`1x1` and :code:`2x2` layouts. Raw formats with
  :code:`1x1` and :code:`2x2` layouts using 1 bit, 6 bit, and 12 bit counter
  depth are supported as well. Support for raw MIB data in other layouts and bit
  depths can be added on demand (:pr:`1169`, :issue:`1135`).
* New attributes :attr:`libertem.udf.base.UDFMeta.sig_slice` and
  :attr:`libertem.udf.base.UDFMeta.tiling_scheme_idx`. These attributes can be used for performant
  access to the current signal slice - mostly important for throughput-limited
  analysis (:pr:`1167`, :issue:`1166`).
* New :code:`--preload` option to :code:`libertem-server` and :code:`libertem-worker`.
  That makes it work as documented in :ref:`hdf5`, following
  `Dask worker preloading
  <https://docs.dask.org/en/stable/how-to/customize-initialization.html#preload-scripts>`_
  (:pr:`1151`).
* Allow selection of I/O backend in GUI and Python API (:issue:`753`, :pr:`896,1129`).
* Re-add support for direct I/O. It was previously only supported as a special
  case for raw files on Linux. Now it is supported for all native dataset
  formats we support on Linux and Windows. Notable exceptions are the OS X
  platform or HDF5, MRC, and SER formats (:pr:`1129`, :issue:`753`).
* Support for reading TVIPS binary files, i.e. :code:`*_NNN.tvips` files (:pr:`1179`).

Bugfixes
--------

* Allow running CoM analysis on a linescan dataset by only returning divergence
  and curl if they are defined (:issue:`1138`, :pr:`1139`).
* :code:`make_dask_array` now works correctly when a :code:`roi` is specified
  (:issue:`933`).
* Correct shape of buffer views in :meth:`~libertem.udf.base.UDFTileMixin.process_tile`
  when the tile has depth 1 (:pr:`1215`).


Documentation
-------------

* Information on multithreading added to UDF docs in :ref:`threading` (:pr:`1170`).

Miscellaneous
-------------

* A `Docker image with a LiberTEM installation
  <https://hub.docker.com/r/libertem/libertem/tags>`_ is available on DockerHub
  now. See :ref:`containers` for details (:pr:`1144`, :issue:`484`).
* Improve performance with large UDF parameters (:pr:`1143`).
* Start using :mod:`libertem.preload` again and import :code:`hdf5plugin` if
  present so that users don't have to specify this common selection of HDF5
  filters as preload themselves (:pr:`1160`).


.. _`v0-8-0`:

0.8.0 / 2021-10-04
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5547992.svg
   :target: https://doi.org/10.5281/zenodo.5547992

This release mainly contains improvements of center of mass / first moment
analysis and support for starting the web GUI from JupyterHub or JupyterLab.

New features
------------

* Support for center of mass with annular masks in :meth:`~libertem.api.Context.create_com_analysis`,
  :class:`~libertem.analysis.com.COMAnalysis` and the GUI (:issue:`633`, :pr:`1089`).
* Support in the GUI for specifying rotation of scan against detector and
  flipping the detector y axis (:pr:`1087`, :issue:`31`). Previously this was only
  supported in the Python API.
* Tweaks and instructions for JupyterHub and JupyterLab integration in LiberTEM, see :ref:`jupyter integration` (:pr:`1074`).
  New package `LiberTEM/LiberTEM-jupyter-proxy <https://github.com/LiberTEM/LiberTEM-jupyter-proxy>`_
  for interfacing.
* In the web API, support was added to re-run visualization only, without
  re-running UDFs for an analysis. This allows for almost instant feedback
  for some operations, like changing CoM parameters.
* Added token-based authentication. For now, it is only usable via
  integrations like Jupyter. It will be extended to local/manual usage
  later (:pr:`1074`, :issue:`1097`). Please comment on :issue:`1097` if local/manual use
  would be beneficial for you so that it is prioritized accordingly.
* SEQ dataset: Added support for loading excluded pixels from XML (:issue:`805`, :pr:`1077`).
  See :class:`~libertem.io.dataset.seq.SEQDataSet` for more information. Also
  support both :code:`*.seq.seq` and :code:`*.seq` as extension for the main SEQ file
  to find files with matching base name that contain correction data (:issue:`1120`, :pr:`1121`).

Bugfixes
--------

* Assert that the :code:`files` argument to :class:`~libertem.io.dataset.dm.DMDataSet` is actually a list or tuple,
  to prevent iterating over a string path (:pr:`1058`).
* Escape globs to support special characters in file names for multi-file
  datasets (:issue:`1066`, :pr:`1067`).
* Make sure multithreading in the main process still works properly after
  launching a :class:`~libertem.api.Context` (:issue:`1053`, :pr:`1100`).
* Allow custom plots to return RGB as plot data, for example a color
  wheel for vector fields (:issue:`1052`, :pr:`1101`).
* Adjust partition count to match the number of CPU compute workers,
  not total workers to prevent residual partitions (:issue:`1086`, :pr:`1103`).
* Correct partition shape for ROI in :class:`~libertem.udf.base.UDFMeta` (:pr:`1109`).
* Fix memory leak: Don't submit dynamically generated callables directly to the distributed cluster,
  as they are cached in an unbounded cache (:issue:`894,964`, :pr:`1119`).

Documentation
-------------

* Note on handling HDF5 files with non-standard compression
  in :class:`~libertem.io.dataset.hdf5.H5DataSet` (:pr:`1059`).
* Link to two more public datasets: :ref:`hires STO` and :ref:`synthetic STO` (:pr:`1073`).

Misc
----

* Speed up coordinate calculation (:issue:`1108`, :pr:`1109`).
* Make sure tasks are scheduled dynamically on available workers if they have uneven
  run time to benefit more from GPUs (:pr:`1107`).
* Cache loaded libraries to reduce overhead of setting the thread count (:issue:`1117`, :pr:`1118`).

Many thanks to our new contributors Levente Puskás for the excluded pixel loading and to
Matthew Bryan for figuring non-standard compression in HDF5 and improving DM
input validation. Congratulations to Alex for closing the long-standing CoM issue :issue:`31`
and for enabling easy and secure access to the web interface on shared IT infrastructure.

.. _`v0-7-1`:

0.7.1 / 2021-07-08
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.5083032.svg
   :target: https://doi.org/10.5281/zenodo.5083032

This is a bugfix release that ensures compatibility with the upcoming numba 0.54
release.

Our custom numba caching makes some assumptions about numba internals, which
have changed in numba 0.54. This fixes compatibility with numba 0.54, and also
makes sure we fail gracefully for future changes (:issue:`1060`, :pr:`1061`).

.. _`v0-7-0`:

0.7.0 / 2021-06-10
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4923277.svg
   :target: https://doi.org/10.5281/zenodo.4923277

This release introduces features that are essential for live data processing,
but can be used for offline processing as well: Live plotting, API for bundled
execution of several UDFs in one run, iteration over partial UDF results, and
asynchronous UDF execution. Features and infrastructure that are specific to
live processing are included in the `LiberTEM-live
<https://github.com/LiberTEM/LiberTEM-live/>`_ package, which will be released
soon.

New features
------------

* Support for postprocessing of results on the main node after merging partial
  results. This adds :meth:`~libertem.udf.base.UDF.get_results` and the
  :code:`use` parameter to :meth:`~libertem.udf.base.UDF.buffer`. See :ref:`udf
  final post processing` for details (:pr:`994`, :pr:`1003`, :issue:`1001`).
* Obtain partial results from each merge step iteratively as a generator
  using :meth:`~libertem.api.Context.run_udf_iter`. See :ref:`partial` and an
  `example
  <https://github.com/LiberTEM/LiberTEM/blob/master/examples/async.ipynb>`_ for
  details (:pr:`1011`)!
* Run multiple UDFs in one pass over a single :code:`DataSet` by passing a
  list of UDFs instead of one UDF in :meth:`~libertem.api.Context.run_udf` and
  :meth:`~libertem.api.Context.run_udf_iter` (:pr:`1011`).
* Allow usage from an asynchronous context with the new :code:`sync=False`
  argument to :meth:`~libertem.api.Context.run_udf` and
  :meth:`~libertem.api.Context.run_udf_iter`. See :ref:`partial` and an `example
  <https://github.com/LiberTEM/LiberTEM/blob/master/examples/async.ipynb>`_ for
  details (:issue:`216`, :pr:`1011`)!
* Live plotting using the new :code:`plots` parameter for
  :meth:`~libertem.api.Context.run_udf` and
  :meth:`~libertem.api.Context.run_udf_iter`, as well as live plotting classes
  documented in :ref:`viz reference`. Pass :code:`plots=True` for simple usage.
  See :ref:`plotting` as well as `an example
  <https://github.com/LiberTEM/LiberTEM/blob/master/examples/live-plotting.ipynb>`_
  for the various possibilities for advanced usage (:issue:`980`, :pr:`1011`).
* Allow some UDF-internal threading. This is mostly
  interesting for ad-hoc parallelization on top of the
  :class:`~libertem.executor.inline.InlineJobExecutor` and live processing that
  currently relies on the :class:`~libertem.executor.inline.InlineJobExecutor`
  for simplicity, but could also be used for hybrid multiprocess/multithreaded
  workloads. Threads for numba, pyfftw, OMP/MKL are automatically
  controlled. The executor makes the number of allowed threads available as
  :attr:`libertem.udf.base.UDFMeta.threads_per_worker` for other threading
  mechanisms that are not controlled automatically (:pr:`993`).
* K2IS: reshaping, sync offset and time series support. Users can now specify a
  :code:`nav_shape`, :code:`sig_shape` and :code:`sync_offset` for a K2IS data
  set, and load time series data (:pr:`1019`, :issue:`911`). Many thanks to
  `@AnandBaburajan <https://github.com/AnandBaburajan>`_ for implementing this
  feature!
* Support for Python >=3.9.3, use Python 3.9 in AppImage (:issue:`914`, :pr:`1037,1039`).

Bugfixes
--------

* UDF: Consistently use attribute access in :code:`UDF.process_*()`, :code:`UDF.merge()`,
  :code:`UDF.get_results()` etc. instead of mixing it with :code:`__getitem__()`
  dict-like access. The previous method still works, but triggers a :class:`UserWarning`
  (:issue:`1000`, :pr:`1003`).
* Also allow non-sliced assignment, for example
  :code:`self.results.res += frame` (:issue:`1000`, :pr:`1003`).
* Better choice of :code:`kind='nav'` buffer fill value outside ROI.

  * String : Was :code:`'n'`, now :code:`''`
  * bool : Was :code:`True`, now :code:`False`
  * integers : Was smallest possible value, now :code:`0`
  * objects : was :code:`np.nan`, now :code:`None` (:pr:`1011`)

* Improve performance for chunked HDF5 files, especially compressed HDF5 files
  which have a chunking in both navigation dimensions. They were causing
  excessive read amplification (:pr:`984`).
* Fix plot range if only zero and one other value are present
  in the result, most notably boolean values (:issue:`944`, :pr:`1011`).
* Fix axes order in COM template: The components in the field are (x, y)
  while the template had them as (y, x) before (:pr:`1023`).

Documentation
-------------

* Update Gatan Digital Micrograph (GMS) examples to work with the current GMS and
  LiberTEM releases and demonstrate the new features. (:issue:`999`,
  :pr:`1002,1004,1011`). Many thanks to Winnie from Gatan for helping to work
  around a number of issues!
* Restructure UDF documentation (:pr:`1034`).
* Document coordinate meta information (:issue:`928`, :pr:`1034`).

Obsolescence
------------

* Removed deprecated blobfinder and :code:`FeatureVecMakerUDF` as
  previously announced. Blobfinder is available as a separate package at
  https://github.com/liberTEM/LiberTEM-blobfinder. Instead of
  :code:`FeatureVecMakerUDF`, you can use a sparse matrix and
  :code:`ApplyMasksUDF` (:pr:`979`).
* Remove deprecated :code:`Job` interface as previously announced.
  The functionality was ported to the more capable UDF interface :pr:`978`.



.. _`v0-6-0`:

0.6.0 / 2021-02-16
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.4543704.svg
   :target: https://doi.org/10.5281/zenodo.4543704

We are pleased to announce the latest LiberTEM release, with many improvements
since 0.5. We would like to highlight the contributions of our GSoc 2020
students `@AnandBaburajan <https://github.com/AnandBaburajan>`_ (reshaping and
sync offset correction) and `@twentyse7en <https://github.com/twentyse7en>`_,
(Code generation to replicate GUI analyses in Jupyter notebooks) who implemented
significant improvements in the areas of I/O and the user interface.

Another highlight of this release is experimental support of NVidia GPUs, both
via CuPy and via native libraries. The API is ready to be used, including support
in the GUI. Performance optimization is still to be done (:issue:`946`).
GPU support is activated for all mask-based analyses (virtual detector and
Radial Fourier) for testing purposes, but will not bring a noticeable
improvement of performance yet. GPU-based processing did show significant
benefits for computationally heavy applications like the SSB implementation in
https://github.com/Ptychography-4-0/ptychography.

A lot of work was done to implement tiled reading, resulting in a
new I/O system. This improves performance in many circumstances, especially
when dealing with large detector frames. In addition, a correction module was
integrated into the new I/O system, which can correct gain, subtract a dark
reference, and patch pixel defects on the fly. See below for the full
changelog!

New features
------------

* I/O overhaul

  * Implement tiled reading for most file formats
    (:issue:`27`, :issue:`331`, :issue:`373`, :issue:`435`).
  * Allow UDFs that implement :code:`process_tile` to influence the tile
    shape by overriding :meth:`libertem.udf.base.UDF.get_tiling_preferences`
    and make information about the tiling scheme available to the UDF through
    :attr:`libertem.udf.base.UDFMeta.tiling_scheme`. (:issue:`554`,
    :issue:`247`, :issue:`635`).
  * Update :code:`MemoryDataSet` to allow testing with different
    tile shapes (:issue:`634`).
  * Added I/O backend selection (:pr:`896`), which allows users to select the best-performing
    backend for their circumstance when loading via the new :code:`io_backend`
    parameter of :code:`Context.load`. This fixes a K2IS performance regression
    (:issue:`814`) by disabling any readahead hints by default. Additionaly, this fixes
    a performance regression (:issue:`838`) on slower media (like HDDs), by
    adding a buffered reading backend that tries its best to linearize I/O per-worker.
    GUI integration of backend selection is to be done.
  * For now, direct I/O is no longer supported, please let us know if this is an
    important use-case for you (:issue:`716`)!

* Support for specifying logging level from CLI (:pr:`758`).
* Support for Norpix SEQ files (:issue:`153`, :pr:`767`).
* Support for MRC files, as supported by ncempy (:issue:`152`, :pr:`873`).
* Support for loading stacks of 3D DM files (:pr:`877`). GUI integration still to be done.
* GUI: Filebrowser improvements: users can star directories in the file browser for easy navigation (:pr:`772`).
* Support for running multiple UDFs "at the same time", not yet exposed in public APIs (:pr:`788`).
* GUI: Users can add or remove scan size dimensions according to the dataset's shape (:pr:`779`).
* GUI: Shutdown button to stop server, useful for example for JupyterHub integration (:pr:`786`).
* Infrastructure for consistent coordinate transforms are added in
  :mod:`libertem.corrections.coordinates` and :mod:`libertem.utils`. See also a
  description of coordinate systems in :ref:`concepts`.
* :meth:`~libertem.api.Context.create_com_analysis` now allows to specify a :code:`flipped y axis`
  and a scan rotation angle to deal with MIB files and scan rotation correctly. (:issue:`325`, :pr:`786`).
* Corrections can now be specified by the user when running a UDF (:pr:`778,831,939`).
* Support for loading dark frame and gain map that are sometimes shipped with SEQ data sets.
* GPU support: process data on CPUs, CUDA devices or both (:pr:`760`, :ref:`udf cuda`).
* Spinning out holography to a separate package is in progress: https://github.com/LiberTEM/LiberTEM-holo/
* Implement CuPy support in :class:`~libertem.udf.holography.HoloReconstructUDF`, currently deactivated due to :issue:`815` (:pr:`760`).
* GUI: Allows the user to select the GPUs to use when creating a new local cluster (:pr:`812`).
* GUI: Support to download Jupyter notebook corresponding to an analysis
  made by a user in GUI (:pr:`801`).
* GUI: Copy the Jupyter notebook cells corresponding to the
  analysis directly from GUI, including cluster connection details (:pr:`862`, :pr:`863`)
* Allow reshaping datasets into a custom shape. The :code:`DataSet` implementations (currently except HDF5 and K2IS)
  and GUI now allow specifying :code:`nav_shape` and :code:`sig_shape`
  parameters to set a different shape than the layout in the
  dataset (:issue:`441`, :pr:`793`).
* All :code:`DataSet` implementations handle missing data
  gracefully (:issue:`256`, :pr:`793`).
* The :code:`DataSet` implementations (except HDF5 and K2IS)
  and GUI now allow specifying a :code:`sync_offset` to
  handle synchronization/acquisition problems (:pr:`793`).
* Users can access the coordinates of a tile/partition slice
  through :attr:`~libertem.udf.base.UDFMeta.coordinates` (:issue:`553`, :pr:`793`).
* Cache warmup when opening a data set: Precompiles jit-ed functions on a single process per node, in a controlled manner,
  preventing CPU oversubscription. This improves further through implementing caching for functions which capture other functions
  in their closure (:pr:`886`, :issue:`798`).
* Allow selecting lin and log scaled visualization for sum, stddev, pick and single mask analyses 
  to handle data with large dynamic range. This adds key :code:`intensity_lin` to
  :class:`~libertem.analysis.sum.SumResultSet`, :class:`~libertem.analysis.sum.PickResultSet`
  and the result of :class:`~libertem.analysis.sd.SDAnalysis`.
  It adds key :code:`intensity_log` to :class:`~libertem.analysis.sum.SingleMaskResultSet`.
  The new keys are chosen to not affect existing keys
  (:issue:`925`, :pr:`929`).
* Tuples can be added directly to :code:`Shape` objects. Right
  addition adds to the signal dimensions of the :code:`Shape`
  object while left addition adds to the navigation
  dimensions (:pr:`749`)

Bugfixes
--------

* Fix an off-by-one error in sync offset for K2IS data (drive-by change in :pr:`706`).
* Missing-directory error isn't thrown if it's due to last-recent-directory not being available (:pr:`748`).
* GUI: when cluster connection fails, reopen form with parameters user submitted (:pr:`735`).
* GUI: Fixed the glitch in file opening dialogue by disallowing parallel browsing before loading is concluded (:pr:`752`).
* Handle empty ROI and extra_shape with zero. Empty result buffers of the appropriate shape are returned if the ROI
  is empty or :code:`extra_shape` has a zero (:pr:`765`)
* Improve internals of :mod:`libertem.io.corrections.detector` and
  :mod:`libertem.io.corrections.corrset` to better support correction
  of many dead pixels. (:pr:`890`, :issue:`889`)
* Handle single-frame partitions in combination with aux data.
  Instead of squeezing the aux buffer, reshape to the correct shape (:issue:`791`, :pr:`902`).
* Libertem-server can now be started from Bash on Windows (:pr:`731`)
* Fix reading without a copy from multi-file datasets. The start offset of the file was
  not taken account when indexing into the memory maps (:issue:`903`).
* Improve performance and reduce memory consumption of point analysis.
  Custom right hand side matrix product to reduce memory consumption and
  improve performance of sparse masks, such as point analysis. See also
  `scipy/13211 <https://github.com/scipy/scipy/issues/13211>`_ (:issue:`917`, :pr:`920`). 
* Fix stability issue with multiple dask clients. :code:`dd.as_completed` needs
  to specify the :code:`loop` to work with multiple :code:`dask.distributed` clients (:pr:`921`).
* GUI: Snap to pixels in point selection analysis. Consistency between point
  selection and picking (:issue:`926`, :pr:`927`).
* Open datasets with autodetection, positional and keyword arguments.
  Handle keyword and positional arguments to :code:`Context.load('auto', ...)`
  correctly (:issue:`936`, :pr:`938`).

Documentation
-------------

* Switched to the readthedocs sphinx theme, improving the overall
  documentation structure. The developer documentation is now in
  a separate section from the user documentation.

Misc
----

* Command line options can also be accessed with shorter alternatives (:pr:`757`).
* Depend on Numba >= 0.49.1 to support setting Numba thread count (:pr:`783`), bumped to 0.51
  to support caching improvements (:pr:`886`).
* libertem-server: Ask for confirmation if the user press ctrl+c. Can immediately stop using
  another ctrl+c (:pr:`781`).
* Included `pytest-benchmark <https://pytest-benchmark.readthedocs.io/en/latest/usage.html>`_
  to integrate benchmarks in the test infrastructure. See :ref:`benchmarking` for details (:pr:`819`).
* The X and Y components for the color wheel visualization in Center of
  Mass and Radial Fourier Analysis are swapped to match the axis convention in
  empyre. This just changes the color encoding in the visualization and not the
  result (:pr:`851`).

Deprecations
------------

* The :code:`tileshape` parameter of :code:`DataSet` implementations is deprecated in
  favor of tileshape negotiation and will be ignored, if given (:issue:`754`, :pr:`777`).
* Remove color wheel code from :code:`libertem.viz` and replace with imports from empyre.
  Note that these functions expect three vector components instead of two (:pr:`851`).
* The new and consistent :code:`nav_shape` and :code:`sig_shape` parameters should be used
  when loading data. The old :code:`scan_size` and :code:`detector_size` parameters,
  where they existed, are still recognized (:pr:`793`).

.. _`v0-5-1`:

0.5.1 / 2020-08-12
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3982290.svg
   :target: https://doi.org/10.5281/zenodo.3982290

Bugfixes
--------

* Allow installation with latest dask distributed on Python 3.6 and 3.7

.. _`v0-5-0`:

0.5.0 / 2020-04-23
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3763313.svg
   :target: https://doi.org/10.5281/zenodo.3763313

New features
------------

* In addition to tuples, :class:`~libertem.common.shape.Shape` objects can be used as
  :code:`extra_shape` parameter for :meth:`libertem.udf.base.UDF.buffer` and
  :meth:`libertem.udf.base.UDF.aux_data` now. (:pr:`694`)
* Progress bar support based on :code:`tqdm` that can be enabled by passing
  :code:`progress=True` to :meth:`libertem.api.Context.run_udf`,
  :meth:`libertem.api.Context.run` and :meth:`libertem.api.Context.map`: :ref:`progress bar`. (:pr:`613,670,655`)
* Include explicit support for Direct Electron's DE5 format based on HDF5. (:pr:`704`)
* GUI: Downloadable results as HDF5, NPZ, TIFF, and RAW. See
  :ref:`download results` for details. (:pr:`665`)
* :meth:`libertem.api.Context.load` now automatically detects file
  type and parameters if :code:`filetype="auto"` is passed. (:pr:`610,621,734`)
* Relocatable GUI: Allow LiberTEM to run from different URL prefixes, allowing integration into,
  for example, JupyterLab. (:pr:`697`)
* Run :meth:`~libertem.udf.base.UDFPreprocessMixin.preprocess` also before merge on
  the main node to allocate or initialize buffers, in addition to running on the
  workers (:pr:`624`).
* No need to set thread count environment variables anymore since the thread count
  for OpenBLAS, OpenMP, Intel MKL and pyFFTW is now set on the workers at run-time.
  Numba support will be added as soon as Numba 0.49 is released. (:pr:`685`).

Bugfixes
--------

* A large number of usability improvements (:pr:`622,639,641,642,659,666,690,699,700,704`).
  Thanks and credit to many new contributors from GSoC!
* Fixed the buggy "enable Direct I/O" checkbox of the RAW dataset and
  handle unsupported operating systems gracefully. (:pr:`696,659`)


Documentation
-------------

* Added screenshots and description of ROI
  and stddev features in usage docs (:pr:`669`)
* Improved instructions for installing LiberTEM
  (general: :pr:`664`; for development: :pr:`598`)
* Add information for downloading and generating sample
  datasets: :ref:`sample data`. (:pr:`650,670,707`)

Obsolescence
------------

* Parameters :code:`crop_detector_to` and :code:`detector_size_raw` of
  :class:`libertem.io.dataset.raw.RawFileDataSet` are deprecated and will be removed
  after 0.6.0. Please specify :code:`detector_size` instead or use a specialized DataSet, for example for EMPAD.
* :class:`libertem.udf.feature_vector_maker.FeatureVecMakerUDF` is deprecated
  and will be removed in 0.6.0. Use :class:`~libertem.udf.masks.ApplyMasksUDF`
  with a sparse stack of single pixel masks or a stack generated by
  :meth:`libertem_blobfinder.common.patterns.feature_vector` instead.
  (:pr:`618`)

Misc
----

* Clustering analysis

  + Use a connectivity matrix to only cluster neighboring pixels,
    reducing memory footprint while improving speed and quality (:pr:`618`).
  + Use faster :class:`~libertem.udf.masks.ApplyMasksUDF` to generate feature
    vector (:pr:`618`).

* :class:`~libertem.udf.stddev.StdDevUDF`

  + About 10x speed-up for large frames (:pr:`625,640`)
  + Rename result buffers of :class:`~libertem.udf.stddev.StdDevUDF`,
    :meth:`~libertem.udf.stddev.run_stddev` and
    :meth:`~libertem.udf.stddev.consolidate_result` from :code:`'sum_frame'` to
    :code:`'sum'`, :code:`'num_frame'` to :code:`'num_frames'` (:pr:`640`)
  + Resolve ambiguity between variance and sum of variances in result buffer names of
    :class:`~libertem.udf.stddev.StdDevUDF`,
    :meth:`~libertem.udf.stddev.run_stddev` and
    :meth:`~libertem.udf.stddev.consolidate_result`. (:pr:`640`)

* LiberTEM works with Python 3.8 for experimental use. A context using a remote Dask.Distributed cluster
  can lead to lock-ups or errors with Python 3.8. The default local Dask.Distributed context works.
* Improve performance with large tiles. (:pr:`649`)
* :class:`~libertem.udf.sum.SumUDF` moved to the :mod:`libertem.udf` folder (:pr:`613`).
* Make sure the signal dimension of result buffer slices can be
  flattened without creating an implicit copy (:pr:`738`, :issue:`739`)

Many thanks to the contributors to this release: :user:`AnandBaburajan`,
:user:`twentyse7en`, :user:`sayandip18`, :user:`bdalevin`, :user:`saisunku`,
:user:`Iamshankhadeep`, :user:`abiB27`, :user:`sk1p`, :user:`uellue`

.. _`v0-4-1`:

0.4.1 / 2020-02-18
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3674003.svg
   :target: https://doi.org/10.5281/zenodo.3674003

This is a bugfix release, mainly constraining the :code:`msgpack` dependency,
as distributed is not compatible to version 1.0 yet. It also contains
important fixes in the HDF5 dataset.

Bugfixes
--------

* Fix HDF5 with automatic tileshape (:pr:`608`)
* Fix reading from HDF5 with roi beyond the first partition (:pr:`606`)
* Add version constraint on msgpack

.. _`v0-4-0`:

0.4.0 / 2020-02-13
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3666686.svg
   :target: https://doi.org/10.5281/zenodo.3666686

The main points of this release are the :ref:`job deprecation` and restructuring
of our packaging, namely :ref:`extracting the blobfinder module <restructuring-0-4>`.

New features
------------

* :code:`dtype` support for UDFs :ref:`udf dtype` (:issue:`549`, :pr:`550`)
* Dismiss error messages via keyboard: allows pressing the escape key to close all currently open error messages (:issue:`437`)
* ROI doesn't have any effect if in pick mode, so we hide the dropdown in that case (:issue:`511`)
* Make tileshape parameter of HDF5 DataSet optional (:pr:`578`)
* Open browser after starting the server. Enabled by default, can be disabled using --no-browser (:issue:`81`, :pr:`580`)
* Implement :class:`libertem.udf.masks.ApplyMasksUDF` as a replacement of ApplyMasksJob (:issue:`549`, :pr:`550`)
* Implement :class:`libertem.udf.raw.PickUDF` as a replacement of PickFrameJob (:issue:`549`, :pr:`550`)
 
Bug fixes
---------

* Fix FRMS6 in a distributed setting. We now make sure to only do I/O in methods that are running on worker nodes (:pr:`531`).
* Fixed loading of nD HDF5 files. Previously the HDF5 DataSet was hardcoded for
  4D data - now, arbitraty dimensions should be supported (:issue:`574`, :pr:`567`)
* Fix :code:`DaskJobExecutor.run_each_host`. Need to pass :code:`pure=False` to ensure multiple runs of the function (:pr:`528`).

Obsolescence
------------

* Because HDFS support is right now not tested (and to my knowledge also not
  used) and the upstream :code:`hdfs3` project is not actively maintained, remove
  support for HDFS. :code:`ClusterDataSet` or :code:`CachedDataSet` should be used
  instead (:issue:`38`, :pr:`534`).

Misc
----

* Depend on distributed>=2.2.0 because of an API change. (:pr:`577`)
* All analyses ported from Job to UDF back-end. The Job-related code remains for now for comparison purposes (:issue:`549`, :pr:`550`)

.. _`job deprecation`:

Job API deprecation
-------------------

The original Job API of LiberTEM is superseded by the new :ref:`user-defined
functions` API with release 0.4.0. See :issue:`549` for a detailed overview
of the changes. The UDF API brings the following advantages:

* Support for regions of interest (ROIs).
* Easier to implement, extend and re-use UDFs compared to Jobs.
* Clean separation between back-end implementation details and application-specific code.
* Facilities to implement non-trivial operations, see :ref:`advanced udf`.
* Performance is at least on par.

For that reason, the Job API has become obsolete. The existing public
interfaces, namely :meth:`libertem.api.Context.create_mask_job` and
:meth:`libertem.api.Context.create_pick_job`, will be supported in LiberTEM for
two more releases after 0.4.0, i.e. including 0.6.0. Using the Job API will
trigger deprecation warnings starting with this release. The new
:class:`~libertem.udf.masks.ApplyMasksUDF` replaces
:class:`~libertem.job.masks.ApplyMasksJob`, and :class:`~libertem.udf.raw.PickUDF`
replaces :class:`~libertem.job.raw.PickFrameJob`.

The Analysis classes that relied on the Job API as a back-end are already ported
to the corresponding UDF back-end. The new back-end may lead to minor
differences in behavior, such as a change of returned dtype. The legacy code for
using a Job back-end will remain until 0.6.0 and can be activated during the
transition period by setting :code:`analysis.TYPE = 'JOB'` before running.

From :class:`~libertem.job.masks.ApplyMasksJob` to :class:`~libertem.udf.masks.ApplyMasksUDF`
.............................................................................................

Main differences:

* :class:`~libertem.udf.masks.ApplyMasksUDF` returns the result with the first
  axes being the dataset's navigation axes. The last dimension is the mask
  index. :class:`~libertem.job.masks.ApplyMasksJob` used to return transposed
  data with flattened navigation dimension.
* Like all UDFs, running an :class:`~libertem.udf.masks.ApplyMasksUDF` returns a
  dictionary. The result data is accessible with key :code:`'intensity'` as a
  :class:`~libertem.common.buffers.BufferWrapper` object.
* ROIs are supported now, like in all UDFs.

.. testsetup:: jobdeprecation

    import numpy as np
    import libertem
    import matplotlib.pyplot as plt

    def all_ones():
        return np.ones((16, 16))

    def single_pixel():
        buf = np.zeros((16, 16))
        buf[7, 7] = 1
        return buf

Previously with :class:`~libertem.job.masks.ApplyMasksJob`:

.. code-block:: python

    # Deprecated!
    mask_job = ctx.create_mask_job(
      factories=[all_ones, single_pixel],
      dataset=dataset
    )
    mask_job_result = ctx.run(mask_job)

    plt.imshow(mask_job_result[0].reshape(dataset.shape.nav))

Now with :class:`~libertem.udf.masks.ApplyMasksUDF`:

.. testcode:: jobdeprecation

    mask_udf = libertem.udf.masks.ApplyMasksUDF(
      mask_factories=[all_ones, single_pixel]
    )
    mask_udf_result = ctx.run_udf(dataset=dataset, udf=mask_udf)

    plt.imshow(mask_udf_result['intensity'].data[..., 0])

From :class:`~libertem.job.raw.PickFrameJob` to :class:`~libertem.udf.raw.PickUDF`
..................................................................................

:class:`~libertem.job.raw.PickFrameJob` allowed to pick arbitrary contiguous
slices in both navigation and signal dimension. In practice, however, it was
mostly used to extract single complete frames.
:class:`~libertem.udf.raw.PickUDF` allows to pick the *complete* signal
dimension from an arbitrary non-contiguous region of interest in navigation
space by specifying a ROI.

If necessary, more complex subsets of a dataset can be extracted by constructing
a suitable subset of an identity matrix for the signal dimension and using it
with ApplyMasksUDF and the appropriate ROI for the navigation dimension.
Alternatively, it is now easily possible to implement a custom UDF for this
purpose. Performing the complete processing through an UDF on the worker nodes
instead of loading the data to the central node may be a viable alternative as
well.

:class:`~libertem.udf.raw.PickUDF` now returns data in the native :code:`dtype`
of the dataset. Previously, :class:`~libertem.job.raw.PickFrameJob` converted to
floats.

Using :meth:`libertem.api.Context.create_pick_analysis` continues to be the
recommended convenience function to pick single frames.

.. _`restructuring-0-4`:

Restructuring into sub-packages
-------------------------------

We are currently restructuring LiberTEM into packages that can be installed and
used independently, see :issue:`261`. This will be a longer process and changes
the import locations.

* `Blobfinder <https://libertem.github.io/LiberTEM-blobfinder/>`_ is the first
  module separated in 0.4.0.
* See :ref:`packages` for a current overview of sub-packages.

For a transition period, importing from the previous locations is supported but
will trigger a :code:`FutureWarning`. See :ref:`show warnings` on how to
activate deprecation warning messages, which is strongly recommended while the
restructuring is ongoing.

.. _`v0-3-0`:

0.3.0 / 2019-12-12
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3572855.svg
   :target: https://doi.org/10.5281/zenodo.3572855

New features
------------

* Make OOP based composition and subclassing easier for
  :class:`~libertem.udf.blobfinder.correlation.CorrelationUDF` (:pr:`466`)
* Introduce plain circular match pattern :class:`~libertem.udf.blobfinder.patterns.Circular` (:pr:`469`)
* Distributed sharded dataset :class:`~libertem.io.dataset.cluster.ClusterDataSet` (:issue:`136`, :issue:`457`)
* Support for caching data sets :class:`~libertem.io.dataset.cached.CachedDataSet`
  from slower storage (NFS, spinning metal) on fast local storage (:pr:`471`)
* :ref:`Clustering` analysis (:pr:`401,408` by :user:`kruzaeva`).
* :class:`libertem.io.dataset.dm.DMDataSet` implementation based on ncempy (:pr:`497`)

  * Adds a new :meth:`~libertem.common.executor.JobExecutor.map` executor primitive. Used to concurrently
    read the metadata for DM3/DM4 files on initialization.
  * Note: no support for the web GUI yet, as the naming patterns for DM file series varies wildly. Needs
    changes in the file dialog.

* Speed up of up to 150x for correlation-based peak refinement in
  :mod:`libertem.udf.blobfinder.correlation` with a Numba-based pipeline (:pr:`468`)
* Introduce :class:`~libertem.udf.blobfinder.correlation.FullFrameCorrelationUDF` which
  correlates a large number (several hundred) of small peaks (10x10) on small
  frames (256x256) faster than
  :class:`~libertem.udf.blobfinder.correlation.FastCorrelationUDF` and
  :class:`~libertem.udf.blobfinder.correlation.SparseCorrelationUDF` (:pr:`468`)
* Introduce :class:`~libertem.udf.UDFPreprocessMixin` (:pr:`464`)
* Implement iterator over :class:`~libertem.analysis.base.AnalysisResultSet` (:pr:`496`)
* Add hologram simulation
  :func:`libertem.utils.generate.hologram_frame` (:pr:`475`)
* Implement Hologram reconstruction UDF
  :class:`libertem.udf.holography.HoloReconstructUDF` (:pr:`475`)

Bug fixes
---------

* Improved error and validation handling when opening files with GUI (:issue:`433,442`)
* Clean-up and improvements of :class:`libertem.analysis.fullmatch.FullMatcher` (:pr:`463`)
* Ensure that RAW dataset sizes are calculated as int64 to avoid integer overflows (:pr:`495`, :issue:`493`)
* Resolve shape mismatch issue and simplify dominant order calculation in Radial Fourier Analysis (:pr:`502`)
* Actually pass the :code:`enable_direct` parameter from web API to the DataSet

Documentation
-------------

* Created :ref:`authorship` (:pr:`460,483`)
* Change management process (:issue:`443`, :pr:`451,453`)
* Documentation for :ref:`crystallinity map` and :ref:`clustering` analysis (:pr:`408` by :user:`kruzaeva`)
* Instructions for profiling slow tests (:issue:`447`, :pr:`448`)
* Improve API reference on Analysis results (:issue:`494`, :pr:`496`)
* Restructure and update the API reference for a number of UDFs and
  other application-specific code (:issue:`503`, :pr:`507,508`)

Obsolescence
------------

* The Job interface is planned to be replaced with an implementation based on UDFs in one of the upcoming releases.

Misc
----

* Split up the blobfinder code between several files to reduce file size (:pr:`468`)

.. _`v0-2-2`:

0.2.2 / 2019-10-14
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3489385.svg
   :target: https://doi.org/10.5281/zenodo.3489385

Point release to fix a number of minor issues, most notably PR :pr:`439` that
should have been merged for version 0.2.

Bug fixes
---------

* Trigger a timeout when guessing parameters for HDF5 takes too long (:issue:`440` , :pr:`449`)
* Slightly improved error and validation handling when opening files with GUI (:commit:`ec74c1346d93eff58d9e2201a7ead5af7aa7cf44`)
* Recognize BLO file type (:issue:`432`)
* Fixed a glitch where negative peak elevations were possible (:pr:`446`)
* Update examples to match 0.2 release (:pr:`439`)

.. _`v0-2-1`:

0.2.1 / 2019-10-07
##################

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3474968.svg
   :target: https://doi.org/10.5281/zenodo.3474968

Point release to fix a bug in the Zenodo upload for production releases.

.. _`v0-2-0`:

0.2.0 / 2019-10-07
##################

This release constitutes a major update after almost a year of development.
Systematic change management starts with this release.

This is the `release message <https://groups.google.com/g/libertem/c/p7MVoVqXOs0/m/vP_tu6K7CwAJ>`_:

User-defined functions
----------------------

LiberTEM 0.2 offers a new API to define a wide range of user-defined reduction
functions (UDFs) on distributed data. The interface and implementation offers a
number of unique features:

* Reductions are defined as functions that are executed on subsets of the data.
  That means they are equally suitable for distributed computing, for interactive
  display of results from a progressing calculation, and for handling live data¹.
* Interfaces adapted to both simple and complex use cases: From a simple map()
  functionality to complex multi-stage reductions.
* Rich options to define input and output data for the reduction functions, which
  helps to implement non-trivial operations efficiently within a single pass over
  the input data.
* Composition and extension through object oriented programming
* Interfaces that allow highly efficient processing: locality of reference, cache
  efficiency, memory handling

Introduction: https://libertem.github.io/LiberTEM/udf.html

Advanced features: https://libertem.github.io/LiberTEM/udf/advanced.html

A big shoutout to Alex (:user:`sk1p`) who developed it! 🏆

¹User-defined functions will work on live data without modification as soon as
LiberTEM implements back-end support for live data, expected in 2020.

Support for 4D STEM applications
--------------------------------

In parallel to the UDF interface, we have implemented a number of applications
that make use of the new facilities:

* Correlation-based peak finding and refinement for CBED (credit: Karina Ruzaeva :user:`kruzaeva`)
* Strain mapping
* Clustering
* Fluctuation EM
* Radial Fourier Series (advanced Fluctuation EM)

More details and examples: https://libertem.github.io/LiberTEM/applications.html

Extended documentation
----------------------

We have greatly improved the coverage of our documentation:
https://libertem.github.io/LiberTEM/index.html#documentation

Fully automated release pipeline
--------------------------------

Alex (:user:`sk1p`) invested a great deal of effort into fully automating our release
process. From now on, we will be able to release more often, including service
releases. 🚀

Basic dask.distributed array integration
----------------------------------------

LiberTEM can generate efficient dask.distributed arrays from all supported
dataset types with this release. That means it should be possible to use our high-performance file
readers in applications outside of LiberTEM.

File formats
------------

Support for various file formats has improved. More details:
https://libertem.github.io/LiberTEM/formats.html

.. _`v0-1-0`:

0.1.0 / 2018-11-06
##################

Initial release of a minimum viable product and proof of concept.

Support for applying masks with high throughput on distributed systems with
interactive web GUI display and scripting capability.
