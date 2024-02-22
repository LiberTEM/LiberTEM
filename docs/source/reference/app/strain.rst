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

.. deprecated:: 0.14.0
   The modules :code:`libertem.analysis.gridmatching` and :code:`libertem.analysis.fullmatch`
   that contain classes and helper functions tp extract and manipulate lattices from correlation results
   are moved to :mod:`libertem_blobfinder.common.gridmatching` and :mod:`libertem_blobfinder.common.fullmatch`,
   and will be removed in a later release.
