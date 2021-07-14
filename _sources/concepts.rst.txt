.. _`concepts`:

Concepts
========

LiberTEM is developed to fulfill the requirements for 4D STEM data processing
and can be used for other processing tasks for n-dimensional detector data as
well.

In 4D STEM, an electron beam is scanned over a sample and for each position,
a 2D image is recorded. That means there are two spatial axes, and two "detector" axes.
We also call the spatial axes "scanning axes" or, more generally, navigation axes. This roughly
corresponds to the batch axis in machine learning frameworks.

The "detector" axes are also called "signal axes", and a single 2D image is also called a frame.

Axis order
----------

We generally follow the numpy convention for axis order, so for a 4D data set,
you could have a :code:`(ny, nx, sy, sx)` tuple describing the shape.

MATLAB users should keep one thing in mind. The navigation axes in Python is the transpose of that of MATLAB. 
In Python, the indices increase linearly with the row. A 3x3 Python matrix is represented in the following way:
 
.. code-block:: python

    [[0,1,2],
    [3,4,5],
    [6,7,8]]
	
`The official "NumPy for Matlab users" documentation`_ might be helpful for Matlab users.

Coordinate system
-----------------

LiberTEM works in pixel coordinates corresponding to array indices. That means
(0, 0) is on the top left corner, the x axis points to the right and the y axis
points to the bottom. This follows the usual plotting conventions for pixel
data.

LiberTEM uses a right-handed coordinate system, which means the z axis points away and positive
rotations are therefore clock-wise.

.. note::
    Please note that the data can be saved with different relation of physical coordinates and
    pixel coordinates. Notably, MIB reference files from Quantum Detectors Merlin cameras have their
    y axis inverted when displayed with LiberTEM. LiberTEM generally
    doesn't deal with such transformations in the numerical back-end.

    In :meth:`~libertem.api.Context.create_com_analysis`, a capability to flip the y axis and rotate
    the shift coordinates is added in version 0.6.0 to support processing MIB files and
    calculate results with physical meaning in electron microscopy, such as the curl and divergence.
    See also :issue:`325`.

    Discussion regarding full support for physical units can be found in :issue:`121`.

Multidimensional data
---------------------

While our GUI is currently limited to 2D visualizations, the Python API does not have that
limitation. You can load data of arbitraty dimensionality and specify an application-specific shape
using the GUI or the Python API, provided our I/O routines support the format. Most of our methods
are currently built for 2D image data, so it should be no problem to load and process for
example time series.

If you want to process data with, for example, 1D or 3D samples, you can write
:ref:`UDFs <user-defined functions>`. Note that in that case, a "frame" is no longer 2D!

.. _The official "NumPy for Matlab users" documentation: https://numpy.org/doc/1.18/user/numpy-for-matlab-users.html#numpy-for-matlab-users
