[Bugfix] Fix loading of malformed EMPAD data
============================================

* For some data sets, the actual raw data acquired corresponds to the
  :code:`mode="search"` scan parameters, and not :code:`mode="acquire"`. This
  is probably a bug in the acquisition software, but we want to be able to load
  these files anyways (:issue:`1617`, :pr:`1620`).
