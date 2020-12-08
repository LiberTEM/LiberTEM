[Bugfix] Improve performance and reduce memory consumption of point analysis
============================================================================

* Custom right hand side matrix product to reduce memory consumption and
  improve performance of sparse masks, such as point analysis. See also
  https://github.com/scipy/scipy/issues/13211 (:issue:`917`, :pr:`920`).