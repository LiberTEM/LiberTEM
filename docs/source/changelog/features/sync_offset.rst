[Feature] Allow setting a sync_offset for datasets
==================================================

* The :code:`DataSet` implementations (except HDF5 and K2IS)
  and GUI now allow specifying a :code:`sync_offset` to
  handle synchronization/acquisition problems (:pr:`793`).
