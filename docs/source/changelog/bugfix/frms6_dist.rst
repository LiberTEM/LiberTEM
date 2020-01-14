[Bugfix] Fix FRMS6 in a distributed setting
===========================================

 * We now make sure to only do I/O in methods that are running on worker nodes (:pr:`531`).

