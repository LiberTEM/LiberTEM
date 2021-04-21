.. _`corrections`:

Corrections
===========

LiberTEM includes correction facilities to substract a dark frame, multiply with a gain map
and patch out defect pixels (dead/hot). These corrections are applied on-the-fly when running
a UDF, both in the GUI and via the Python API.

The following data set formats ship with some of
this correction data:

* FRMS6: dark frame is loaded from the first part of the data set
* SEQ: dark frame and gain map are loaded from MRC sidecar files (:code:`<basename>.dark.mrc` and :code:`<basename>.gain.mrc`)

In the GUI, all corrections that are supplied by the data set will be applied. In the Python API,
the user can decide to pass their own corrections to apply, via the :code:`corrections` parameter
of :code:`Context.run` and :code:`Context.run_udf`. It expects a :class:`~libertem.corrections.CorrectionSet` object, which
can also be empty to disable corrections completely. For example:

.. testsetup:: *

    import numpy as np
    from libertem.executor.inline import InlineJobExecutor
    from libertem.udf.sum import SumUDF
    from libertem import api

    ctx = api.Context(executor=InlineJobExecutor())
    data = np.random.random((16, 16, 32, 32)).astype(np.float32)
    dataset = ctx.load("memory", data=data, sig_dims=2)

.. testcode::

    from libertem.corrections import CorrectionSet
    import sparse

    # excluded pixels are passed as a sparse COO matrix, which can be built
    # in different ways, here is one way:
    excluded = np.zeros((32, 32), dtype=bool)
    excluded[5, 16] = 1
    excluded = sparse.COO(excluded)

    ctx.run_udf(udf=SumUDF(), dataset=dataset, corrections=CorrectionSet(
        dark=np.zeros((32, 32)),
        gain=np.ones((32, 32)),
        excluded_pixels=excluded,
    ))


.. autoclass:: libertem.corrections.corrset.CorrectionSet
