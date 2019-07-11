Strain mapping
==============

LiberTEM can evaluate the position of convergent beam electron diffraction disks or precession electron diffraction peaks to generate input data for strain maps in three automated steps:

1. Identify peaks using :meth:`~libertem.udf.blobfinder.get_peaks`
2. Extract a probable lattice from the peak positions using :meth:`~libertem.analysis.fullmatch.full_match`
3. Refine the lattice for each frame using :meth:`~libertem.udf.blobfinder.run_refine`

The algorithms are currently focused on the initial data extraction step, i.e. they work purely in pixels in frame coordinates and derive only parameters for each individual peak and for a 2D lattice in the detector reference frame. They don't try to index peaks in crystallographic indices of the sample or to derive a 3D orientation. However, they can extract relevant input data such as peak positions and intensities very efficiently for such subsequent processing steps.

The algorithms are designed to be robust against intensity variations across a diffraction disk, between disks and between frames by calculating a correlation for each potential peak position, measuring the quality of each correlation and using the quality and position with subpixel refinement in a weighted least square optimization to derive the parameters for each frame. At the same time, the algorithms are optimized for efficiency and scalability so that they generate a full strain map on 32 GB of raw data within a few minutes on a suitable workstation and scale on a cluster, making full use of LiberTEM's distributed processing capabilities.

Relevant input parameters are

* Type and parameters for the diffraction disk
    * Background subtraction with positive center and negative ring for small, uniform disks such as diffraction patterns recorded with precession
        * Parameters are inner and outer radius
    * Radial gradient for larger disks with internal structure, i.e. CBED
        * Parameter is radius
    * Custom matching template. See for example :meth:`~libertem.masks.radial_gradient_background_subtraction`
* Number of disks to find in the initial step
* Padding to define the search area around the expected position
* Tolerance for position errors in the matching routine

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

Reference
~~~~~~~~~

.. automodule:: libertem.udf.blobfinder
   :members:
   :undoc-members:
   :special-members: __init__

.. automodule:: libertem.analysis.gridmatching
   :members:
   :undoc-members:
   :special-members: __init__

.. automodule:: libertem.analysis.fullmatch
   :members:
   :undoc-members:
   :special-members: __init__
