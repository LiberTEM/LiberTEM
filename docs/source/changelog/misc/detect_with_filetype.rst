[Misc] Dataset detection uses file suffix if possible
=====================================================

* The function :code:`detect` to automatically determine dataset type
  will now use the file suffix as a hint to choose its search order.
  This may lead to faster responses in the web client when configuring
  a new dataset. As an additional feature, :code:`MemoryDataSet` will
  now be auto-detected without error. (:pr:`1377`)
