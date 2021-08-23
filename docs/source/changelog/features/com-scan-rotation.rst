[Feature] GUI Support for advanced parameters in CoM / first moment
===================================================================

* Support in the GUI for specifying rotation of scan against detector and
  flipping the detector y axis (pr:`1087`, :issue:`31`)
* In the web API, support was added to re-run visualization only, without
  re-running UDFs for an analysis. This allows for almost instant feedback
  for some operations, like changing CoM parameters
