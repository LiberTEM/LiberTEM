[Bugfix] Fix :code:`KeyError` after exception
=============================================

* After a consumer of :code:`PipelinedExecutor.run_tasks` throws an exception,
  it's possible that results from the old run still end up in the response queue.
  Log them as a warning and ignore (:pr:`1308`)
