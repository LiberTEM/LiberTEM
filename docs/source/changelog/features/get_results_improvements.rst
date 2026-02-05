[Feature] Consistent access to results
======================================
* In case of async iteration over results, we now make sure the merge function
  is not running concurrently with accessing and using the results. Also, the
  :meth:`~libertem.udf.base.UDF.get_results` method is called lazily, reducing
  overheads (:pr:`1632`).

