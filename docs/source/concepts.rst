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

While our GUI is currently limited to 4D data sets, the Python API does not
have that limitation. You can load data of arbitraty dimensionality, provided our I/O
routines support the format. Most of our methods are currently built for 2D image data,
so it should be no problem to load and process for example time series.

If you want to process data with, for example, 1D or 3D samples, you can write
:ref:`UDFs <user-defined functions>`. Note that in that case, a "frame" is no longer 2D!
