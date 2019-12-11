.. _`application api`:

Application-specific API
========================

.. toctree::
    :maxdepth: 2

    app/amorphous

This section documents application-specific code.

Correlation-based peak finding and strain mapping
-------------------------------------------------

.. _`blobfinder api`:

Blobfinder
~~~~~~~~~~

This module contains classes and helper functions that allow to apply correlation and refinement on a dataset
using the :ref:`user-defined functions` interface.

.. automodule:: libertem.udf.blobfinder
   :members:
   :special-members: __init__

.. _`matching api`:

Matching
~~~~~~~~

These modules contain classes and helper functions that extract and manipulate lattices from correlation results.

.. automodule:: libertem.analysis.gridmatching
   :members:
   :undoc-members:
   :inherited-members:
   :special-members: __init__

.. automodule:: libertem.analysis.fullmatch
   :members:
   :undoc-members:
   :inherited-members:
   :special-members: __init__
