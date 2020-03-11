Changelog
=========

.. testsetup:: *

    from libertem import api
    from libertem.executor.inline import InlineJobExecutor

    ctx = api.Context(executor=InlineJobExecutor())
    dataset = ctx.load("memory", datashape=(16, 16, 16, 16), sig_dims=2)

.. _continuous:
.. _`v0-5-0`:

0.5.0.dev0 (continuous)
#######################

.. toctree::
   :glob:

   changelog/*/*

.. _latest:
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
functions` API with release 0.4.0.dev0. See :issue:`549` for a detailed overview
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

.. testcode:: jobdeprecation

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
  module separated in 0.4.0.dev0.
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
    * Adds a new :meth:`~libertem.executor.base.JobExecutor.map` executor primitive. Used to concurrently
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

This is the `release message <https://groups.google.com/d/msg/libertem/p7MVoVqXOs0/vP_tu6K7CwAJ>`_: 

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
