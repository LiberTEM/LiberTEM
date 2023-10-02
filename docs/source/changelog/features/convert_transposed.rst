[Feature] Converter for transposed DM4 datasets
===============================================
* A function :function:`libertem.contrib.convert_transposed.convert_dm4_transposed`
  has been added to efficiently convert Gatan Digital Micrograph STEM datasets
  stored in :code:`(sig, nav)` ordering to numpy .npy files in :code:`(nav_sig)`
  ordering (:pr:`1509`).
