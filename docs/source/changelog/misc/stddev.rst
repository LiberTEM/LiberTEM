[Misc] Rename result buffers of StdDevUDF
=========================================

* Rename result buffers of :class:`~libertem.udf.stddev.StdDevUDF`,
  :meth:`~libertem.udf.stddev.run_stddev` and
  :meth:`~libertem.udf.stddev.consolidate_result` from :code:`'sum_frame'` to
  :code:`'sum'`, :code:`'num_frame'` to :code:`'num_frames'` (:pr:`640`)
* Resolve ambiguity between variance and sum of variances in result buffer names of
  :class:`~libertem.udf.stddev.StdDevUDF`,
  :meth:`~libertem.udf.stddev.run_stddev` and
  :meth:`~libertem.udf.stddev.consolidate_result`. (:pr:`640`)
