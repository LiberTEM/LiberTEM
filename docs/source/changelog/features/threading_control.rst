[Feature] Allow some UDF-internal threading
===========================================
* The executor can now allow UDFs to perform some internal threading,
  making the number of allowed threads available as :code:`UDFMeta.threads_per_worker`.
  This is mostly interesting for ad-hoc parallelization on top of the
  :code:`InlineJobExecutor`, but could also be used for hybrid
  multiprocess/multithreaded workloads (:pr:`993`).
  Threads for numba, pyfftw, OMP/MKL are still automatically controlled,
  the :code:`UDFMeta.threads_per_worker` addition is meant for other threading mechanisms.
