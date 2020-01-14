[Bugfix] Fix DaskJobExecutor.run_each_host
==========================================

 * Need to pass :code:`pure=False` to ensure multiple runs of the function (:pr:`528`).
