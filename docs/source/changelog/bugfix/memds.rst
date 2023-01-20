[Bugfix] Consistent shape parameter handling in :class:`libertem.io.dataset.memory.MemoryDataSet`
=================================================================================================

* The handling of :code:`sync_offset`, :code:`nav_shape`, :code:`sig_shape` and
  :code:`sig_dims` in :class:`libertem.io.dataset.memory.MemoryDataSet` is now
  consistent (:pr:`1207`). 