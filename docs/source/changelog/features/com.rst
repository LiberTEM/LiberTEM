[Feature] Coordinate transforms for center of mass analysis
===========================================================

* :meth:`libertem.api.Context.create_com_analysis` now allows to specify a flipped y axis
  and a scan rotation angle to deal with MIB files and scan rotation correctly. (:issue:`325`, :pr:`786`)