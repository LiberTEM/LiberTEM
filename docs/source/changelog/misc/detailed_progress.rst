[Misc] Progress bar can provide frame-level updates
===================================================

* The :code:`progress=True` feature of :code:`ctx.run()`
  and :code:`ctx.run_udf()` has been improved to provide updates
  on a frame-by-frame basis during a run, in the case of
  slow-running analyses/UDFs. The number of partitions currently
  being processed is now also displayed in parentheses when known.
  Progress bars in a Jupyter notebook can now also render as
  Javascript widgets where available via :code:`tqdm.auto`, where
  the :code:`ipywidgets` package is installed.
  (:pr:`1341`)
