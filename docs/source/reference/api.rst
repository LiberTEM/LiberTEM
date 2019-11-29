LiberTEM Context API
--------------------

See :ref:`api documentation` for introduction and complete examples.

Context
~~~~~~~

.. automodule:: libertem.api
   :members:
   :special-members: __init__

Job API
~~~~~~~

.. automodule:: libertem.job.base
   :members:
   :special-members: __init__

MaskContainer and ApplyMaskjob
..............................

:class:`libertem.job.masks.MaskContainer` and :class:`libertem.job.masks.ApplyMasksJob` allow to implement
highly efficient mask application operations, such as virtual detector, center of mass or
feature vector calculations. In the future, they might be migrated to the :ref:`user-defined functions` API.

.. automodule:: libertem.job.masks
   :members:
   :special-members: __init__

PickFrameJob
............

.. automodule:: libertem.job.raw
   :members:
   :special-members: __init__

Analysis API
~~~~~~~~~~~~

.. automodule:: libertem.analysis.base
   :members:
   :undoc-members:
   :special-members: __init__

.. automodule:: libertem.analysis.com
   :members:
   :special-members: __init__

.. automodule:: libertem.analysis.disk
   :members:
   :special-members: __init__

.. automodule:: libertem.analysis.ring
   :members:
   :special-members: __init__

.. automodule:: libertem.analysis.point
   :members:
   :special-members: __init__

.. automodule:: libertem.analysis.sum
   :members:
   :special-members: __init__

.. automodule:: libertem.analysis.raw
   :members:
   :special-members: __init__
