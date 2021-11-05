Improve performance with large parameters
=========================================

* Previously, parameters were baked into the :code:`UDFTask` objects, so they were
  transferred multiple times for a single UDF run. To allow for larger parameters,
  they are now handled separately from the function that is run (:pr:`1143`).
