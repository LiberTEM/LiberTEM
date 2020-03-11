[Bugfix] Performance improvements of clustering analysis
========================================================

* Use a connectivity matrix to only cluster neighboring pixels,
  reducing memory footprint while improving speed and quality (:pr:`618`)
* Use faster :class:`~libertem.udf.masks.ApplyMasksUDF` to generate feature vector (:pr:`618`)