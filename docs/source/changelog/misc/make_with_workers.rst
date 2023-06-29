[Misc] Context.make_with can specify workers
============================================

* The :code:`Context.make_with` constructor method has been improved
  to allow simple specification of the number CPU and GPU workers to use,
  as well as the type of executor to create, where such a specification is
  supported (:pr:`1443`).
