[Obsolescence] Remove HDFS support
==================================

 * Because HDFS support is right now not tested (and to my knowledge also not
   used) and the upstream :code:`hdfs3` project is not actively maintained, remove
   support for HDFS. :code:`ClusterDataSet` or :code:`CachedDataSet` should be used
   instead (:issue:`38`, :pr:`534`).
