[Feature] I/O Overhaul
======================

* Implement tiled reading for most file formats
  (:issue:`27`, :issue:`331`, :issue:`373`, :issue:`435`)
* Allow UDFs that implement :code:`process_tile` to influence the tile
  shape and make information about the tiling scheme available to the UDF
  (:issue:`554`, :issue:`247`, :issue:`635`)
* Update :code:`MemoryDataSet` to allow testing with different
  tile shapes (:issue:`634`)
* Added I/O backend selection (:pr:`896`), which allows users to select the best-performing
  backend for their circumstance when loading via the new :code:`io_backend`
  parameter of :code:`Context.load`. This fixes a K2IS performance regression
  (:issue:`814`) by disabling any readahead hints by default. Additionaly, this fixes
  a performance regression (:issue:`838`) on slower media (like HDDs), by
  adding a buffered reading backend that tries its best to linearize I/O per-worker.
* For now, direct I/O is no longer supported, please let us know if this is an
  important use-case for you!
