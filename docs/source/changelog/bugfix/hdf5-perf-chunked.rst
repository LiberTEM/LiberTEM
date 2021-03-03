[Bugfix] Improve performance for chunked HDF5 files
===================================================
* Especially compressed HDF5 files, which have a chunking in both navigation
  dimensions were causing excessive read amplification (:pr:`984`)
