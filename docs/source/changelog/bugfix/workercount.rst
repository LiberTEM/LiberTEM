[Bugfix] Calculate worker count for partitioning correctly
==========================================================

* Only count CPU workers to make sure we don't have residual
  partitions (:issue:`1086`, :pr:`1103`).
