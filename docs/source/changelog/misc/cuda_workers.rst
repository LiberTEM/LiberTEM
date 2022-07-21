[Misc] Allow spawning multiple CUDA workers on same device
==========================================================

* :func:`~libertem.executor.dask.cluster_spec` now accepts the same
  CUDA device ID multiple times to spawn multiple workers on the same GPU.
  This can help increase GPU resource utilisation for some workloads.