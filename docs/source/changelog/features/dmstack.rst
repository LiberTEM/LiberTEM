[New] Support for stacks of DM3/DM4 files
=========================================

* :class:`libertem.io.dataset.dm.DMDataSet` implementation based on ncempy.
* Adds a new :meth:`~libertem.executor.base.JobExecutor.map` executor primitive. Used to concurrently
  read the metadata for DM3/DM4 files on initialization.
* Note: no support for the web GUI yet, as the naming patterns for DM file series varies wildly. Needs
  changes in the file dialog.
