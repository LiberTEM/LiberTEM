[Bugfix] Support raw_csr with indptr of dtype uint64
====================================================

* Avoid upcasting to float64 with uint64 by downcasting to int64 before calculations (:pr:`1465`).
