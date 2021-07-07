[Bugfix] Fix numba 0.54 compatibility
=====================================

* Our custom numba caching makes some assumptions about numba internals, which
  have changed in numba 0.54. This fixes compatibility with numba 0.54, and also
  makes sure we fail gracefully for future changes (:issue:`1060`).
