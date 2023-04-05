[Misc] Allow progress bar redirection
=====================================

* The :code:`progress=True` parameter of :code:`ctx.run()`,
  :code:`ctx.run_udf()` and similar methods on :code:`Context`
  now accepts an instance of :code:`ProgressReporter`, added in
  :pr:`1341`, to allow redirecting progress messages to another
  endpoint.
