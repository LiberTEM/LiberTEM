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

We generally follow the numpy convention for axis order, so for a 4D data set,
you could have a :code:`(ny, nx, sy, sx)` tuple describing the shape.

MATLAB users should keep one thing in mind. The navigation axes in Python is the transpose of that of MATLAB. 
In Python, the indices increase linearly with the row. A 3x3 Python matrix is represented in the following way:
 
.. code-block:: python

    [[0,1,2],
    [3,4,5],
    [6,7,8]]
	
`The official "NumPy for Matlab users" documentation`_ might be helpful for Matlab users.

While our GUI is currently limited to 4D data sets, the Python API does not
have that limitation. You can load data of arbitraty dimensionality, provided our I/O
routines support the format. Most of our methods are currently built for 2D image data,
so it should be no problem to load and process for example time series.

If you want to process data with, for example, 1D or 3D samples, you can write
:ref:`UDFs <user-defined functions>`. Note that in that case, a "frame" is no longer 2D!

.. _The official "NumPy for Matlab users" documentation: https://numpy.org/doc/1.18/user/numpy-for-matlab-users.html#numpy-for-matlab-users
