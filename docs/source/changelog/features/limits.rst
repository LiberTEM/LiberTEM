[Feature] Reject outliers in plots
==================================

* This change implements an outlier filter for live plots and the web interface.
  It excludes values from the plot range calculation that are at the very edge
  of the value distribution and very different from the remaining values. In
  particular, this fixes plots of Dectris detector data where their acquisition
  software sets dead pixels to the maximum possible value. This improvement is
  particularly important for the web interface and for live plots since they
  currently do not allow setting the plot range manually. The value zero is
  excluded from the statistical evaluation to avoid cutting off the few non-zero
  pixels in low-dose datasets (:issue:`1310` :pr:`1742`).
