[Bugfix] fix compatibility with the latest matplotlib
=====================================================
* Applying a :code:`Norm` to >2D data is no longer allowed, so we reshape
  the data appropriately. Also needed an additional fix for all-nan results (:issue:`1007`, :pr:`1008`).
