
.. _`io backends`:

I/O Backends
============

By default, on Windows, a buffered strategy is chosen, whereas Linux and Mac
OS use unbuffered memory-mapped I/O. You can pass an instance of
:class:`~libertem.io.dataset.base.backend.IOBackend` as the :code:`io_backend`
parameter of :meth:`~libertem.api.Context.load` to use a different backend.
This allows you to override the default backend choice, or set parameters
for the backend.

Available I/O backends
----------------------

BufferedBackend
~~~~~~~~~~~~~~~

.. autoclass:: libertem.io.dataset.base.BufferedBackend
    :noindex:

MmapBackend
~~~~~~~~~~~

.. autoclass:: libertem.io.dataset.base.MMapBackend
    :noindex:
