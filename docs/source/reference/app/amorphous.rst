Amorphous materials reference
=============================

.. note::

    See :ref:`amorphous app` for an overview and description of the amorphous applications.

.. _`fem api`:

Fluctuation EM
--------------

This module contains the UDF for applying FEM to a single ring (mostly useful for interactive use).

.. automodule:: libertem.udf.FEM
   :members:
   :exclude-members: get_result_buffers, get_task_data
   :special-members: __init__

.. _`radial fourier api`:

Radial Fourier Analysis
-----------------------

This module contains the radial fourier series analysis, for analysing frequencies and
symmetries of diffraction patterns.

.. automodule:: libertem.analysis.radialfourier
   :members: RadialFourierResultSet
   :special-members: __init__
