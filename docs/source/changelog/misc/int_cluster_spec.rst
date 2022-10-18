[Misc] Specify workers using integer arguments
==============================================

* The methods :method:`PipelinedExecutor.make_spec` and
  :function:`libertem.executor.dask.cluster_spec` both now
  accept integers for their :code:`cpus` and :code:`cudas`
  arguments, in addition to the existing iterable forms.
  (:issue:`1294`).
