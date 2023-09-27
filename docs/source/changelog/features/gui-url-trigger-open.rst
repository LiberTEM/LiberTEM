[Feature] Browse or open data sets via URL hash parameters
==========================================================
* You can now specify a path via a URL fragment, which will be opened when
  LiberTEM is ready for loading data. For example like this:
  http://localhost:9000/#action=open&path=/path/to/your/data/
  The path can either point to a file or a directory, triggering the file browser
  or directly the open dialog (:issue:`1085`, :pr:`1518`).

