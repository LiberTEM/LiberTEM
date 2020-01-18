.. _`udf reference`:

UDF API reference
-----------------

Defining UDFs
~~~~~~~~~~~~~

See :ref:`user-defined functions` for an introduction and in-depth explanation.

.. automodule:: libertem.udf
   :members:
   :special-members: __init__
   :exclude-members: UDFTask,UDFRunner

Running UDFs
~~~~~~~~~~~~

Two methods of :class:`libertem.api.Context` are relevant for running user-defined functions:

.. autoclass:: libertem.api.Context
   :members: run_udf,map
   :noindex:

Buffers
~~~~~~~

:class:`~libertem.common.buffers.BufferWrapper` objects are used to manage data in the context of user-defined functions.

.. automodule:: libertem.common.buffers
   :members:
   :undoc-members:
   :special-members: __init__

.. _`utilify udfs`:

Included utility UDFs
~~~~~~~~~~~~~~~~~~~~~

Some generally useful UDFs are included with LiberTEM:

.. note::
    See :ref:`application api` for application-specific UDFs and analyses.

.. autoclass:: libertem.udf.logsum.LogsumUDF
    :members:
    :special-members: __init__

.. autoclass:: libertem.udf.stddev.StdDevUDF
    :members:
    :special-members: __init__

.. autoclass:: libertem.udf.sumsigudf.SumSigUDF
    :members:
    :special-members: __init__

.. autoclass:: libertem.udf.masks.ApplyMasksUDF
    :members:
    :special-members: __init__

.. autoclass:: libertem.udf.raw.PickUDF
    :members:
    :special-members: __init__
