[Bugfix] Reset environment properly when setting thread count
=============================================================

* Make sure the environment is cleaned up properly in the main process
  after adjusting it temporarily to launch Dask workers with thread
  count settings (:issue:`1053`, :pr:`1100`)