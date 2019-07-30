Phase-change materials
======================
Phase-change materials are some of the most promising materials for data-storage applications. In this application section, the pixelated STEM dataset with the AgInSbTe phasechange material specimen will be processed.
Distinguishing between amorphous and crystalline regions can be hard in microstructure analysis with 4-D-STEM because of low contrast between them during two most common method of visualization: bright and dark field imaging.
The main difference of pixelated diffraction patterns for each of the beam positions the presence of additional (non zero-order) diffraction peaks in crystalline frames, while the amorphous frames have only the zero-order peak.
Detection of the positions of additional peaks for all frames of crystalline regions will allow to gain information on the crystal structure.

Methods to distinguish crystalline and amorphous regions in phase-change materials.

Crystallinity map
~~~~~~~~~~~~~~~~~
First, two different types of the regions, crystalline (low resistance) and amorphous (high resistance), should be distinguished. Crystalline regions are characterized by the presence of additional non-zero
diffraction peaks on diffraction image, while the diffraction image (the frame) of amorphous regions contains only the zero-order diffraction peak.
The presence of additional periodic diffraction peaks leads to differences of the Fourier spectrum of the crystalline frames compared to amorphous frames in the intermediate frequency range.Integration over a ring of all frames and using the result of the integration of the frames spectra over a disk as an intensity for each position for sample visualization
allows to distinguish amorphous and crystalline regions of sample.

GUI use:
--------

You can select "FFT Fourier" from the "Add Analysis" menu in the GUI to obtain crystallinity map with high contrast between crystalline and amorphous regions.

.. figure:: path
Use the checkbox in the midle if you would like to enable/disable of masking of zero-order peak out.

.. figure:: path

In case of enabling zero-order diffraction peak removal, Use the controls on the middle to position the disk-shaped selector over the region of average frameaverage you'd like to mask out.

.. figure:: path

Use the controls on the left to position the ring-shaped selector over the region of frames spectra average you'd like to integrate over and then click "Apply". 

.. figure:: path

Tipps:

In case of sharp and bright zero order peak you will need to mask it out (spectrum of such a peak will be a 2d sinc function which will spread all over the frames spectra)
in case of sinc shaped zero order peak you do not need to remove it. the influence of it will be presented in low frequency range and will be masked out during integration region choosing

.. figure:: path

.. figure:: path

Clustering
~~~~~~~~~~