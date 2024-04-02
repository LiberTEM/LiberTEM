[Feature] Valid data masks
==========================

* Add endpoints to provide :meth:`~libertem.udf.base.UDF.merge`
  and :meth:`~libertem.udf.base.UDF.get_results` with information
  on which parts of the dataset have already been processed
  (:issue:`1473`). The information is then propagated to the result
  buffers, and UDFs can override the validity mask for results
  using the new :meth:`~libertem.udf.base.UDF.with_mask` method
  (:issue:`1449` :pr:`1593`).
