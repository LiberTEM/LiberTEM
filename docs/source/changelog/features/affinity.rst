[Feature] Set CPU affinity when starting dask workers
=====================================================
* This can improve cache locality and thus performance (by a few percent for our tests, :pr:`995`)
