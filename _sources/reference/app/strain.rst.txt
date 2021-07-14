Correlation-based peak finding and strain mapping reference
===========================================================

.. _`blobfinder api`:

Blobfinder
----------

.. deprecated:: 0.4.0
    Blobfinder has moved to its own package LiberTEM-blobfinder with a new
    structure. Please see
    https://libertem.github.io/LiberTEM-blobfinder/index.html for installation
    instructions and documentation of the new structure. Imports from
    :code:`libertem.udf.blobfinder` trigger a :code:`FutureWarning` starting from
    0.4.0 and are supported until LiberTEM release 0.6.0.

.. _`matching api`:

Matching
--------

These modules contain classes and helper functions that extract and manipulate lattices from correlation results.

.. automodule:: libertem.analysis.gridmatching
   :members:
   :show-inheritance:

.. automodule:: libertem.analysis.fullmatch
   :members:
   :show-inheritance:
