[Bugfix] Avoid deadlock on missing tasks
========================================

* Don't wait infinitely for missing in-flight tasks to avoid deadlocks.
