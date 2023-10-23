.. _`as a service`:

LiberTEM as a service
=====================

Opening data via the URL
------------------------

.. versionadded:: 0.13.0

You can specify a path via a URL fragment, which will be opened when the
LiberTEM GUI is ready for loading data. The path can either point to a file or
a directory, triggering the file browser or directly the open dialog. For
example:

* Link to a directory: http://localhost:9000/#action=open&path=/data/some-dir/
* Link to a data set: http://localhost:9000/#action=open&path=/data/some-dir/default.hdr

The motivation for this feature is integration of LiberTEM into data management
systems and electronic lab notebooks, without launching a separate instance for
each data set, for example via :ref:`jupyter integration`.
