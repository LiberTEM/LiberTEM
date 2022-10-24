[Bugfix] Gracefully handle startup timeout
==========================================

* In the pipelined executor, increase default timeout and emit a more user-friendly error message in case of hitting the timeout (:pr:`1342`).
