[Feature] Descan error compensation with :class:`~libertem.udf.masks.ApplyShiftedMasksUDF`
==========================================================================================

* Allow specifying per-frame shifts of the optical axis to compensate descan error
  for all masks-based analyses. The new :class:`libertem.udf.masks.ApplyShiftedMasksUDF`
  works as a drop-in replacement for :class:`libertem.udf.masks.ApplyMasksUDF` in most cases.

