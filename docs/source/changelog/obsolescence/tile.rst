[Removed] Tile metadata attributes
==================================

* Instead of :class:`~libertem.io.dataset.base.tiling.DataTile` objects, an UDF's
  processing method will receive plain array objects, such as
  :class:`numpy.ndarray`, :class:`sparse.SparseArray` etc. That means the
  :code:`scheme_idx` and :code:`tile_slice` attributes are not available from
  the tile anymore, but only from the corresponding
  :attr:`libertem.udf.base.UDFMeta.tiling_scheme_idx` and
  :attr:`libertem.udf.base.UDFMeta.slice`. This change makes handling different
  array types such as sparse arrays or CuPy arrays easier. For CuPy arrays this
  was already the previous behavior.
