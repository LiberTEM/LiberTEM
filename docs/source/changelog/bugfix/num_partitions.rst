[Bugfix] Match partition number to cores
========================================

* Try to split work evenly on all cores to reduce "stragglers" that
  drive up the clock time to finish (:pr:`1796`).