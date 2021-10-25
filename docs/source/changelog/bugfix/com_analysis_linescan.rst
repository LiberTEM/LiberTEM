Fix running CoM analysis on a linescan dataset
==============================================

* CoM analyses implicitly calculate x- and y- gradients of the centre
  positions through np.gradient. This previously failed for nav
  dimensions of length 1. The new behaviour is to only return div/curl
  results in the COMResultSet when they are defined.
  (:issue:`1138`, :issue:`1139`)