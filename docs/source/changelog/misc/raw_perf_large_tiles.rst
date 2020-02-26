[Misc] Improve performance with large tiles
===========================================

* When reading with large tiles from raw files with a roi,
  there were some unnecessary and slow allocations triggered (:pr:`649`)
