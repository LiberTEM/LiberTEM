[Misc] `nav_shape` parameter on HDF5Dataset
===========================================

* The :code:`HDF5Dataset` now supports a :code:`nav_shape`
  argument allowing the data to be reshaped into a different
  scan grid. This is only currently possible where the dataset
  in the file and the desired :code:`nav_shape` contain the
  same number of frames. (:issue:`441`)
