[Feature] Gracefully handle UDF cancellation
============================================

* Allow the internals to raise
  :class:`~libertem.common.executor.JobCancelledError`, which is translated to
  a :class:`~libertem.exceptions.UDFRunCancelled` exception for the user
  (:pr:`1448`).
