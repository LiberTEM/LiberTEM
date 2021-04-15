[Bug] Buffer fill value outside of ROI
======================================

* Better choice of :code:`kind='nav'` buffer fill value outside ROI.
  String: 'n' -> ''; bool: True -> False,
  integers: smallest possible value -> 0,
  objects: np.nan -> None (:pr:`1011`)
