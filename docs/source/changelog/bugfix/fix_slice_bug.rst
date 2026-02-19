[Bugfix] Fix origin length mismatch in :code:`Slice.adjust_for_roi`
===================================================================

* Fixed an origin length mismatch in :meth:`~libertem.common.slice.Slice.adjust_for_roi` caused by negative slicing when ``sig_dims`` is 0 (:pr:`1799`).
