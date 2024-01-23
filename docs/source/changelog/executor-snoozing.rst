[Feature] Free resources after inactivity
=========================================

* :code:`libertem-server` has a new command line option,
  :code:`--snooze-timeout`, which allows to free up resources
  (main memory, GPU memory, CPU time) after a duration of inactivity
  (:issue:`1570`, :pr:`1572`). This is especially useful when deploying
  LiberTEM as a service on a multi-user system.
