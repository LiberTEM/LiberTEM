[Feature] Allow overriding the number of partitions
===================================================
* Adds a parameter :code:`num_partitions` to most :code:`DataSet`
  implementations. This means (expert) users can override the number of
  partitions when loading data, in cases where the default heuristic doesn't
  work well. Also changes the number of partitions in case there are fewer
  frames than workers, where it is an overall advantage to have small (1-frame)
  partitions instead of aggregating all frames into a single partition,
  especially if there is a lot of processing done per frame (:issue:`1701`
  :pr:`1702`).
