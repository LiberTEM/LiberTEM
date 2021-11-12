Re-add support for direct I/O
=============================

* Direct I/O was previously only supported as a special case for raw files,
  now it is supported for all native dataset formats we support - notable
  exceptions are  HDF5, MRC, and SER (:pr:`1129`, :issue:`716`).
