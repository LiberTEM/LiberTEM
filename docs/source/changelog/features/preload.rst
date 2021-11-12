Add :code:`--preload` option to :code:`libertem-server` and :code:`libertem-worker`
===================================================================================

* Make it work as documented in `HDF5 docs
  <https://libertem.github.io/LiberTEM/reference/dataset.html#hdf5>`_, follow
  `Dask worker preloading
  <https://docs.dask.org/en/stable/how-to/customize-initialization.html#preload-scripts>`_
  (:pr:`1151`).