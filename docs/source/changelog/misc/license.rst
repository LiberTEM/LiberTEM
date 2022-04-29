[Misc] Moved file around to allow clean MIT licensing of IO code
================================================================

* Make sure that :mod:`libertem.io` and :mod:`libertem.common` only depend on code
  that is compatible with the MIT license. This required moving
  some code. It is re-imported at the same positions as before to keep backwards
  compatibility. (:issue:`1031`, :pr:`1245`).
