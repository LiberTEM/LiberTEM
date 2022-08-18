[Bugfix] Fix worker killing on Python 3.6
=========================================

* Python 3.6 doesn't have :code:`Process.kill`, emulate using
  :code:`TerminateProcess` / :code:`os.kill` (:pr:`1319`).
* Fix logic that determines when to call :code:`Process.kill`.
