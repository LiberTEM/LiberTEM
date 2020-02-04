[Bugfix] loading of nD HDF5 files
=================================

* Previously the HDF5 DataSet was hardcoded for 4D data - now, arbitraty
  dimensions should be supported (:issue:`574`, :pr:`567`)
