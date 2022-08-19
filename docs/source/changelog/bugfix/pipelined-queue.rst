[Bugfix] :meth:`~libertem.common.executor.SimpleMPWorkerQueue.put_nocopy`
=========================================================================

* Make sure :meth:`~libertem.common.executor.SimpleMPWorkerQueue.put_nocopy`
  puts cloudpickled content that
  :meth:`~libertem.common.executor.SimpleMPWorkerQueue.get()` expects
  (:pr:`1318`).