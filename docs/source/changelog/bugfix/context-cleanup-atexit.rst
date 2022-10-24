[Bugfix] Context cleanup at process exit
========================================

* Automatically close :code:`Context` at process exit, so we don't run code when the interpreter is possibly in half-torn-down status (:pr:`1343`).

