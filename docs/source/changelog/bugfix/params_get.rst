[Bugfix] Fix UDFData.get(key, default)
======================================

* Fix for the method :code:`UDFData.get(key, default)` which previously
  could not return the default parameter due to catching the wrong
  error. (:issue:`1284`).
