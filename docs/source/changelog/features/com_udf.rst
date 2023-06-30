[Features] Centre-of-Mass calculation UDF
=========================================

* Adds COMUDF to perform virtual CoM calculations via
  the standard `run_udf` interface. This replicates the
  existing :code:`Context.create_com_analysis` functionality
  but without the additional overhead constraints of analyses,
  meaning CoM can now be performed during live processing
  or subclassed to perform integrated CoM
  as in `https://github.com/LiberTEM/LiberTEM-iCoM`_
  (:pr:`1392`).
