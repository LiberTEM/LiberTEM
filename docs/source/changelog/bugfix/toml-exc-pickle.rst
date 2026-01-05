[Bugfix] :code:`TOMLDecodeError` cannot be pickled anymore
==========================================================
* In recent tomli versions, we can't rely on pickling the exception instance,
  so we need to wrap it into our own (simpler) type.
