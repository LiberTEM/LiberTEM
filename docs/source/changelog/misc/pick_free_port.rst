[Misc] Server launcher picks free port if default in use
========================================================

* The function launching the `libertem-server` application will now
  pick a free, random port if the default port is in use. Choosing 
  an explicit port will continue to raise an error.
  Closes :issue:`1184`.
