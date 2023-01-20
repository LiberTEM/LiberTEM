[Bugfix] Handle empty partition results due to :code:`sync_offset`
==================================================================

* Don't divide by zero when merging an empty partition result in
  :class:`libertem.udf.stddev.StdDevUDF`. This can happen when a partition is
  skipped entirely.