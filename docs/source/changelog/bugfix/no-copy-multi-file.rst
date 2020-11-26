[Bugfix] Fix reading without a copy from multi-file datasets
============================================================

* The start offset of the file was not taken account when indexing
  into the memory maps (:issue:`903`)
