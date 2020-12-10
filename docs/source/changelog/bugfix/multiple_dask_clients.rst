[Bugfix] Fix stability issue with multiple dask clients
=======================================================

* `dd.as_completed` needs to specify the `loop` to work with multiple
  `dask.distributed` clients (:pr:`921`).
