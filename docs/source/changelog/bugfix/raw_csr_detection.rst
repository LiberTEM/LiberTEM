[Bugfix] RawCSR dataset no longer tries to load large files
===========================================================

* The :code:`RawCSRDataSet.detect_params` function will
no longer try to load any file as binary and parse it to TOML,
which could previously lead to large files being loaded during
:code:`ctx.load('auto', path)`. (:pr:`1404`).
