[Feature] Executors and Dask integration
========================================

* Add :class:`~libertem.executor.delayed.DelayedJobExecutor`,
  :class:`~libertem.executor.concurrent.ConcurrentJobExecutor` and
  :meth:`libertem.executor.integration.get_dask_integration_executor`.
* Add :meth:`libertem.api.Context.make_with` to make common executor choices more
  accessible to users.
* Allow using an existing Dask Client as well as setting the LiberTEM Client as default
  Dask scheduler for integration purposes.
* Add :meth:`libertem.contrib.daskadapter.task_results_array` to create
  chunked Dask arrays from intermediate UDF task results.
* Restructure and extend documentation of executors.
* :pr:`1158`, :issue:`1146,922`
