.. _`strain mapping`:

Strain mapping
==============

.. note::

    See :ref:`blobfinder API reference <blobfinder api>` and :ref:`matching API reference <matching api>` for API references

LiberTEM can evaluate the position of convergent beam electron diffraction disks or precession electron diffraction peaks to generate input data for strain maps in three automated steps:

1. Identify peaks using :meth:`~libertem.udf.blobfinder.get_peaks`
2. Extract a probable lattice from the peak positions using :meth:`~libertem.analysis.fullmatch.FullMatcher.full_match`
3. Refine the lattice for each frame using :meth:`~libertem.udf.blobfinder.run_refine`

The algorithms are currently focused on the initial data extraction step, i.e. they work purely in pixels in frame coordinates and derive only parameters for each individual peak and for a 2D lattice in the detector reference frame. They don't try to index peaks in crystallographic indices of the sample or to derive a 3D orientation. However, they can extract relevant input data such as peak positions and intensities very efficiently for such subsequent processing steps.

The algorithms are designed to be robust against intensity variations across a diffraction disk, between disks and between frames by calculating a correlation for each potential peak position, measuring the quality of each correlation and using the quality and position with subpixel refinement in a weighted least square optimization to derive the parameters for each frame. At the same time, the algorithms are optimized for efficiency and scalability so that they generate a full strain map on 32 GB of raw data within a few minutes on a suitable workstation and scale on a cluster, making full use of LiberTEM's distributed processing capabilities.

Relevant input parameters are

* Matching template
    * Instance of :class:`~libertem.udf.blobfinder.MatchPattern`
    * Available options are :class:`~libertem.udf.blobfinder.RadialGradient`, :class:`~libertem.udf.blobfinder.BackgroundSubtraction`, :class:`~libertem.udf.blobfinder.RadialGradientBackgroundSubtraction`, and :class:`~libertem.udf.blobfinder.UserTemplate`
    * :code:`search` parameter to define the search area around the expected position
* Matcher: Instance of :class:`~libertem.analysis.gridmatching.Matcher`.
    * :code:`tolerance` for position errors in the matching routine
* Number of disks to find in the initial step

This example shows a typical strain map of a transistor with strained silicon.

.. toctree::

   strainmap-SiGe

Strain map of polycrystalline materials
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

LiberTEM allows to segment a scan into regions of interest to process grains with different orientations and lattice parameters separately. This example uses clustering of a feature vector to process a polycrystalline sample in a fully automated fashion.

.. toctree::

   strainmap-poly

Acknowledgments
~~~~~~~~~~~~~~~

Karina Ruzaeva implemented the correlation routines and introduced feature vectors and clustering. Alexander Clausen developed the architecture, in particular the interface for user-defined functions that allows to implement such complex processing schemes on a distributed system easily and at the same time with optimal performance. Dieter Weber implemented the grid matching and refinement code.

.. rubric:: Reference

See :ref:`blobfinder API reference <blobfinder api>` and :ref:`matching API reference <matching api>` for details!
