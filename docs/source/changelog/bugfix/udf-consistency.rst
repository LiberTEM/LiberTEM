[Bugfix] UDF: consistency fixes
===============================
* Consistently use attribute access in :code:`UDF.process_*`, :code:`UDF.merge`,
  :code:`UDF.get_results` etc. instead of mixing it with :code:`__getitem__` dict-like
  access (:issue:`1000`, :pr:`1003`)
