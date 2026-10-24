[Bugfix] Ignore data after the last frame with the mmap backend
===============================================================

* Opening a file with extra data after the last frame, such as the tags at
  the end of a DM file or a footer in a raw file, could fail with
  :code:`AssertionError: len(mem) must fit the dtype` when the file size in
  bytes was divisible by the number of pixels per frame. The extra data is
  now always cut off.
