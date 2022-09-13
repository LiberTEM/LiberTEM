[Misc] Optimise Slice/Shape behaviour for tiles
===============================================

* Reduces overhead of (re-)constructing :code:`Slice` and :code:`Shape`
  objects when slicing tiles, in particular focused on the method
  :code:`BufferWrapper.get_contiguous_view_for_tile()`. (:issue:`1313`)
