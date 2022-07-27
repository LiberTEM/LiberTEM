[Bugfix] Correct type determination in :class:`~libertem.udf.auto.AutoUDF`
==========================================================================

* Use the input dtype and not the dataset native dtype to determine UDF output
  shape and dtype for :class:`~libertem.udf.auto.AutoUDF` (:pr:`1298`).
