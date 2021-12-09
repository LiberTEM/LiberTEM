Add :code:`UDFMeta.sig_slice` and :code:`UDFMeta.tiling_scheme_idx`
===================================================================

* These attributes can be used for performant access to the current signal
  slice - mostly important for throughput-limited analysis (:pr:`1167`, :issue:`1166`).
