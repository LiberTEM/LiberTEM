[Bugfix] Fix ri handle bug in Annular CoM Web UI
================================================

* Fix for the web UI entering into a bad state when switching
  between Disk and Annular CoM modes, where the inner radius could
  be greater than the outer radius leading to an all-zeros mask
  (:pr:`1278`).

