[Bugfix] Handle invalid inputs for numWorker
============================================

* Raise if user input for :code:`numWorker` is non-positive integer
  or other error is encountered in :code:`DaskJobExecutor` creation
  from web interface. (:pr:`1334`).
