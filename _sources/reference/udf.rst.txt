.. _`udf reference`:

UDF API reference
-----------------

.. _`define udf ref`:

Defining UDFs
~~~~~~~~~~~~~

See :ref:`user-defined functions` for an introduction and in-depth explanation.

.. automodule:: libertem.udf.base
   :members:
   :exclude-members: Task,UDFTask,UDFRunner

.. _`run udf ref`:

Running UDFs
~~~~~~~~~~~~

Three methods of :class:`libertem.api.Context` are relevant for running user-defined functions:

.. autoclass:: libertem.api.Context
   :members: run_udf,run_udf_iter,map
   :noindex:

.. _`buffer udf ref`:

Buffers
~~~~~~~

:class:`~libertem.common.buffers.BufferWrapper` objects are used to manage data in the context of user-defined functions.

.. automodule:: libertem.common.buffers
   :members:
   :undoc-members:

.. _`utilify udfs`:

Included utility UDFs
~~~~~~~~~~~~~~~~~~~~~

Some generally useful UDFs are included with LiberTEM:

.. note::
    See :ref:`application api` for application-specific UDFs and analyses.

.. _`sum udf`:

Sum of frames
#############

.. autoclass:: libertem.udf.sum.SumUDF
    :members:

.. _`logsum udf`:

Sum of log-scaled frames
########################

.. autoclass:: libertem.udf.logsum.LogsumUDF
    :members:

.. _`stddev udf`:

Standard deviation
##################

.. autoclass:: libertem.udf.stddev.StdDevUDF
    :members:

.. autofunction:: libertem.udf.stddev.run_stddev

.. autofunction:: libertem.udf.stddev.consolidate_result

.. _`sumsig udf`:

Sum per frame
#############

.. autoclass:: libertem.udf.sumsigudf.SumSigUDF
    :members:

.. _`masks udf`:

Apply masks
###########

.. autoclass:: libertem.udf.masks.ApplyMasksUDF
    :members:

.. _`pick udf`:

Load data
#########

.. autoclass:: libertem.udf.raw.PickUDF
    :members:
