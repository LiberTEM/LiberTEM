Fix running CoM analysis on a linescan dataset
==============================================

* CoM analyses implicitly calculate x- and y- gradients of the centre
  positions through np.gradient. This previously failed for nav
  dimensions of length 1. The new behaviour is to return div/curl
  results filled with zeros and valid results otherwise.
  (:issue:`1138`)