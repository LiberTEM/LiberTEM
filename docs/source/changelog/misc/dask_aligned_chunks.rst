[Misc] Dask arrays created via mask_dask_array are rechunk-free
===============================================================

* Modified :code:`libertem.contrib.daskadapter.mask_dask_array` such that
  the dask arrays generated from DataSets should not incurr any
  rechunk operations when they are converted into multi-dimensional
  arrays from their flat navigation dimension form.