[Misc] HDF5Dataset navgiation reshape and sync_offset
=====================================================

* The :code:`HDF5Dataset` now supports a :code:`nav_shape`
  argument allowing the data to be reshaped into a different
  scan grid. Additionally the :code:`sync_offset` parameter is now
  supported to correct for scan/acquisition synchronisation.
  (:issue:`441`, :pr:`1364`)
