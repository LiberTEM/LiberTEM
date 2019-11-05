[New] Full-frame correlation
============================

* Introduce :class:`libertem.udf.blobfinder.FullFrameCorrelationUDF` which
  correlates a large number (several hundred) of small peaks (10x10) on small
  frames (256x256) faster than
  :class:`libertem.udf.blobfinder.FastCorrelationUDF` and
  :class:`libertem.udf.blobfinder.SparseCorrelationUDF`
