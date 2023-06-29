[bugfix] Properly forward log level to workers at init
======================================================

* Previously, the :code:`'silence_logs'` option supplied to
  the :code:`cluster_spec` function was not properly influencing
  the log level on the Dask workers spawned by :code:`DaskJobExecutor`.
  This resulted in a substantial amount of logging information
  being printed to :code:`stderr` at cluster startup. This has been
  worked around with a temporary environment variable (:pr:`1438`).  
