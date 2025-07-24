[Feature] Enable CuPy on main process
=====================================

* Make GPU processing available for operations on the main node (chiefly
  :meth:`libertem.udf.base.UDF.merge` and
  :meth:`libertem.udf.base.UDF.get_results`) through
  :attr:`libertem.udf.base.UDF.xp`. This is targeted at methods like iDPC and
  iCoM where the result is calculated in
  :meth:`libertem.udf.base.UDF.get_results` with Fourier transforms over the
  displacement vector field that run much faster on GPUs compared to CPUs. It
  can be controlled with the new :code:`main_process_gpu` parameter for
  :meth:`libertem.api.Context.make_with`. It is enabled by default to catch
  potential issues, to be reviewed before the next release (:pr:`1759`).
