[Feature] :code:`preprocess()` and AUX data
===========================================

* Run :meth:`~libertem.udf.base.UDFPreprocessMixin.preprocess` before merge on
  the master node to allocate or initialize buffers (:pr:`624`).
* Set correct view of AUX data during
  :meth:`~libertem.udf.base.UDFPreprocessMixin.preprocess` on the control node
  (:pr:`624`)