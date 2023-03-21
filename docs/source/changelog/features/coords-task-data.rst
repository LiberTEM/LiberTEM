[Feature] Make `self.meta.coordinates` available in `UDF.get_task_data`
=======================================================================

* This is needed, for example, if you want to pre-allocate a buffer that is
  partition-like-shaped, but you don't want this buffer to be part of the
  result. A concrete example appears when working in a compressed space, where
  you might want to batch the actual computation for the whole compressed
  partition (:pr:`1397`).
