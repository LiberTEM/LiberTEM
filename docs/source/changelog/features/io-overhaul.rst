[Feature] I/O Overhaul
======================

* Implement tiled reading for most file formats
  (:issue:`27`, :issue:`331`, :issue:`373`, :issue:`435`)
* Allow UDFs that implement :code:`process_tile` to influence the tile
  shape and make information about the tiling scheme available to the UDF
  (:issue:`554`, :issue:`247`, :issue:`635`)
* Update :code:`MemoryDataSet` to allow testing with different
  tile shapes (:issue:`634`)
