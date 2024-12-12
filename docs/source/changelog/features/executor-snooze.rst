[Feature] Executor snooze after inactivity
==========================================
* Adds support for automatically freeing resources used by
  an executor after a period of inactivity. This was previously
  available in the web client, but has now been exposed in the
  Python API. Supported only in the default :code:`DaskJobExecutor`
  at this time (:issue:`1576`, :pr:`1690`).
