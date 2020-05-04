[Bugfix] Handle empty ROI and extra_shape with zero
===================================================

* Empty result buffers of the appropriate shape are returned if the ROI is empty or extra_shape has a zero (:pr:`765`)
