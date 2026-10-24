[Bugfix] Raise on unsupported DM data types
===========================================

* :class:`~libertem.io.dataset.dm_single.SingleDMDataSet` now raises a
  :code:`DataSetException` for DM data types that can't be converted to a
  numpy dtype, such as binary (bool) or packed complex images. Before, these
  were silently read as :code:`uint32`, which gave wrong values.
