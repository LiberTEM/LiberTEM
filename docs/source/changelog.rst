Changelog
=========

.. _continuous:

.. Other parts of Continuous section commented out because of no entries yet
.. .. _`v0-4-0`:

.. 0.4.0.dev0 (continuous)
.. #######################

.. .. toctree::
   :glob:

..   changelog/*/*

.. _latest:
.. _`v0-3-0`:

0.3.0
#####

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

* Split up the bobfinder code between several files to reduce file size (:pr:`468`)

.. _`v0-2-2`:

0.2.2
#####

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

0.2.1
#####

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.3474968.svg
   :target: https://doi.org/10.5281/zenodo.3474968

Point release to fix a bug in the Zenodo upload for production releases.

.. _`v0-2-0`:

0.2.0
#####

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

0.1.0
#####

Initial release of a minimum viable product and proof of concept.

Support for applying masks with high throughput on distributed systems with
interactive web GUI display and scripting capability.
