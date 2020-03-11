Off-axis electron holography
============================

.. versionadded:: 0.3.0

The off-axis holography applications (see :ref:`holography app` for the application examples) are realized in
two modules: UDF for off axis electron holography reconstruction and utility function for hologram simulations.

Hologram reconstruction
-----------------------

The reconstruction module contains class for reconstruction of off-axis holograms using Fourier-space method
which implies following processing steps:

* Fast Fourier transform
* Filtering of the sideband in Fourier space and cropping (if applicable)
* Centering of the sideband
* Inverse Fourier transform.

.. automodule:: libertem.udf.holography
   :members:
   :undoc-members:
   :special-members: __init__

Hologram simulation
-------------------

.. automodule:: libertem.utils.generate
   :members: hologram_frame