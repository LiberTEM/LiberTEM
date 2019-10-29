[New] Full-frame correlation
============================

* Introduce :class:`libertem.udf.blobfinde.FullFrameCorrelationUDF` which correlates a large
  number (several hundred) of small peaks (10x10) on small frames (256x256) faster than
  :class:`libertem.udf.blobfinde.FastCorrelationUDF`
  and :class:`libertem.udf.blobfinde.SparseCorrelationUDF`