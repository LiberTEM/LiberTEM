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

.. deprecated:: 0.4.0
    Use :ref:`user-defined functions` instead. See also :ref:`job deprecation`.

.. automodule:: libertem.job.base
   :members: Job
   :special-members: __init__

Analysis API
~~~~~~~~~~~~

.. automodule:: libertem.analysis.base
   :members: Analysis, AnalysisResult, AnalysisResultSet
   :special-members: __init__

.. automodule:: libertem.analysis.masks
   :members: MasksResultSet, SingleMaskResultSet
   :special-members: __init__

.. automodule:: libertem.analysis.com
   :members: COMResultSet
   :special-members: __init__

.. automodule:: libertem.analysis.sum
   :members: SumResultSet
   :special-members: __init__

.. automodule:: libertem.analysis.raw
   :members: PickResultSet
   :special-members: __init__
