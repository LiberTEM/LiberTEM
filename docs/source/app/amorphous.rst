Amorphous materials
===================

Methods to determine the local order or crystallinity of amorphous and nanocrystalline materials.

Fluctuation EM
~~~~~~~~~~~~~~

TODO

Radial Fourier Series
~~~~~~~~~~~~~~~~~~~~~

Fluctuation EM doesn't evaluate the spatial distribution of intensity. It only works if enough intensity accumulates in each pixel so that local ordering leads to larger intensity variations than just random noise, i.e. if statistical variations from shot noise average out and variations introduced by the sample dominate.

If most detector pixels are hit with one electron or none, the standard deviation between detector positions in a region of interest is the same, even if pixels that received electrons are spatially close, while other regions received no intensity. That means Fluctuation EM doesn't produce contrast between amorphous and crystalline regions if the detector has a high resolution, and/or if only low intensity is scattered into the region of interest.

The Radial Fourier Series Analysis solves this problem by calculating a Fourier series in a ring-shaped region instead of just evaluating the standard deviation. The angle of a pixel relative to the user-defined center point of the diffraction pattern is used as a phase angle for the Fourier series.

Since diffraction patterns usually show characteristic symmetries TODO cite, the strength of the Fourier coefficients of orders 2, 4 and 6 highlight regions with crystalline order for even the lowest intensities. With the relationship between variance in real space and power spectral density in frequency space, the sum of all coefficients that are larger than zero is equivalent to the standard deviation, i.e. Fluctuation EM. Only summing coefficients from lower orders corresponds Fluctuation EM on a smoothened dataset.

Usage:
------

TODO