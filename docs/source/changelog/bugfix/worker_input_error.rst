[Bugfix] Handle invalid inputs for numWorker
============================================

* Raise if user input for `numWorker` is non-positive integer
  or other error is encountered in `DaskJobExecutor` creation
  from web interface. (:pr:`1334`).
