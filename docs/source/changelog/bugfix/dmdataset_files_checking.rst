[Bugfix] Add checking for iterable files= argument to DMDataset
===============================================================

* Add a line checking that the files argument to DMDataset is actually a list or tuple, to prevent iterating over a string path
