[Bugfix] Use int64 for dataset size calculation
===============================================

* Ensure that dataset sizes are calculated as int64 to avoid integer overflows (:pr:`495`, :issue:`493`)