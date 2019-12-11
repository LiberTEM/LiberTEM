Correlation-based peak finding and strain mapping reference
===========================================================

.. _`blobfinder api`:

Blobfinder
----------

This module contains classes and helper functions that allow to apply correlation and refinement on a dataset
using the :ref:`user-defined functions` interface.

Match patterns
~~~~~~~~~~~~~~

Correlation pattern classes with support for Fourier-based fast correlation.

.. automodule:: libertem.udf.blobfinder.patterns
   :members: 
   :special-members: __init__

Correlation
~~~~~~~~~~~

UDFs and utility functions to find peaks and refine their positions by using
correlation.

.. automodule:: libertem.udf.blobfinder.correlation
   :members:
   :show-inheritance:
   :special-members: __init__

Refinement
~~~~~~~~~~

UDFs and utility functions to refine grid parameters from peak positions.

.. automodule:: libertem.udf.blobfinder.refinement
   :members:
   :show-inheritance:
   :special-members: __init__

Blobfinder utilities
~~~~~~~~~~~~~~~~~~~~

General utility functions for the blobfinder module.

.. automodule:: libertem.udf.blobfinder.utils
   :members:
   :show-inheritance:
   :special-members: __init__


.. _`matching api`:

Matching
--------

These modules contain classes and helper functions that extract and manipulate lattices from correlation results.

.. automodule:: libertem.analysis.gridmatching
   :members:
   :show-inheritance:
   :special-members: __init__

.. automodule:: libertem.analysis.fullmatch
   :members:
   :show-inheritance:
   :special-members: __init__
