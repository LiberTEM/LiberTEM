[Bugfix] Prevent locking up file detection
==========================================

* Only attempt opening as HDF5 during autodetection if file magic
  recognizes the file as HDF5 since the HDF5 libray locks up on some
  files (:issue:`1231`, :pr:`1232`).
