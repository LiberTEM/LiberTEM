[Misc] Set thread count at run time
===================================

* No need to set thread count environment variables anymore since the thread count
  for OpenBLAS, OpenMP, Intel MKL and pyFFTW is now set on the workers at run-time.
  Numba support will be added as soon as Numba 0.49 is released. (:pr:`685`).