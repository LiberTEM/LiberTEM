Off-axis electron holography
----------------------------

This module contains class for reconstruction of off-axis holograms using Fourier-space method which implies
following processing steps:

* Fast Fourier transform
* Filtering of the sideband in Fourier space and cropping (if applicable)
* Centering of the sideband
* Inverse Fourier transform.

.. automodule:: libertem.udf.holography
   :members:
   :undoc-members:
   :inherited-members:
   :special-members: __init__