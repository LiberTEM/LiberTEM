[Feature] Support for sparse ROI formats
========================================

* The :code:`roi` argument to methods such as :code:`ctx.run_udf()`
  now supports additional modes, including specifying a single coordinate
  to process (e.g. :code:`(3, 5)`), as well as sparse array formats
  from both :code:`pydata.sparse` and `scipy.sparse`.  (:pr:`1306`).
