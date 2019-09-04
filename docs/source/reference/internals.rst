Internal API
------------

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

MaskContainer and ApplyMaskjob
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`libertem.job.masks.MaskContainer` and :class:`libertem.job.masks.ApplyMaskJob` allow to implement
highly efficient mask application operations, such as virtual detector, center of mass or
feature vector calculations. In the future, they might be migrated to the :ref:`user-defined functions` API.

.. automodule:: libertem.job.masks
   :members: MaskContainer,ApplyMasksJob
   :undoc-members:
   :special-members: __init__