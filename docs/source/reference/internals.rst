Internal API
------------

MaskContainer
~~~~~~~~~~~~~

:class:`libertem.common.container.MaskContainer` helps to implement highly efficient
mask application operations, such as virtual detector, center of mass or feature
vector calculations.

.. versionchanged:: 0.4.0
    Moved from :mod:`libertem.job.masks` to :mod:`libertem.common.container` to
    use it in UDFs and prepare deprecation of the Job interface.

.. automodule:: libertem.common.container
   :members: MaskContainer
   :special-members: __init__

Shapes and slices
~~~~~~~~~~~~~~~~~

These classes help to manipulate shapes and slices of n-dimensional binary data to
facilitate the MapReduce-like processing of LiberTEM. See :ref:`concepts` for
a high-level introduction.

.. automodule:: libertem.common.shape
   :members:
   :undoc-members:
   :special-members: __init__

.. automodule:: libertem.common.slice
   :members:
   :undoc-members:
   :special-members: __init__
