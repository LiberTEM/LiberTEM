[Feature] Allow reshaping datasets into a custom shape
======================================================

* The :code:`DataSet` implementations (except HDF5 and K2IS)
  and GUI now allow specifying :code:`nav_shape` and :code:`sig_shape`
  parameters to set a different shape than the layout in the
  dataset (:issue:`441`, :pr:`793`).
* All :code:`DataSet` implementations handle missing data
  gracefully (:issue:`256`, :pr:`793`).
