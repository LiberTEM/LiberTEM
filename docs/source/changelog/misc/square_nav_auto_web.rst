[Misc] Guess a square nav_shape in the web UI if no other information
=====================================================================

* If a dataset does not declare its navigation shape then guess
  a square nav_shape in the Web UI if the number of detected frames
  is a square number. This can occur for the MIB, TVIPS, SEQ, MRC
  and K2IS dataset formats. (:issue:`1309`)
