[Feature] Improved scheduling for the pipelined executor
========================================================

* Schedule tasks on the worker with the smallest request queue size (:pr:`1451`).
* Update :class:`~libertem.executor.pipelined.PipelinedExecutor` to properly
  match tasks to workers based on resources requested. This means
  UDFs requiring/supporting CUDA or CuPy will now correctly be run on dedicated
  workers (:pr:`1453`).
