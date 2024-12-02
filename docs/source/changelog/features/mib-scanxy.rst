[Feature] MIB: read :code:`ScanX` and :code:`ScanY` from header
===============================================================
* Starting with version 1.5 of the merlin software, the
  `ScanX` and `ScanY` fields are included with the acquisition
  header (:code:`.hdr` file), if the software knows about them.
  We now try to read these and fall back to the old logic
  if they are not present (:issue:`1630` :pr:`1631`).
