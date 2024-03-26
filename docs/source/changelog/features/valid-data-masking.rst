[Feature] Valid data masks
==========================

* Make the mask of already processed navigation elements available
  to the UDF in :meth:`~libertem.udf.base.UDF.merge` and
  :meth:`~libertem.udf.base.UDF.get_results` (:issue:`1473`),
  propagate this information to the result buffers
  and allow UDFs to override this for results using the new
  :meth:`~libertem.udf.base.UDF.with_mask` method
  (:issue:`1449` :pr:`1593`).
