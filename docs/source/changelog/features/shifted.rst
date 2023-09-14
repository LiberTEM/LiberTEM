[Feature] Descan error compensation with :class:`~libertem.udf.masks.ApplyMasksUDF`
===================================================================================

* Allow specifying shifts of the optical axis to compensate descan error
  for all masks-based analyses. The existing :class:`libertem.udf.masks.ApplyMasksUDF`
  was extended to accepts a :code:`shifts` parameter which can define shifts to apply,
  either as a constant for all frames or on a per-frame basis.
