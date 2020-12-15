[Feature] Lin and log scaling for analyses for sum, stddev, pick and single mask
================================================================================

* Allow selecting lin and log scaled visualization for sum, stddev, pick and single mask analyses 
  to handle data with large dynamic range. This adds key :code:`intensity_lin` to
  :class:`~libertem.analysis.sum.SumResultSet`, :class:`~libertem.analysis.sum.PickResultSet`
  and the result of :class:`~libertem.analysis.sd.SDAnalysis`.
  It adds key :code:`intensity_log` to :class:`~libertem.analysis.sum.SingleMaskResultSet`.
  The new keys are chosen to not affect existing keys.
  (:issue:`925`, :pr:`929`)
