.. _`progress reference`:

Progress bar support
--------------------

The :code:`ProgressReporter` is used to display the progress
bar during calls to :class:`libertem.api.Context`. In normal
usage there is no need to instantiate these classes directly,
but they can be used to modify how progress is displayed in
particular applications.


.. autoclass:: libertem.common.progress.ProgressReporter
    :members:

.. autoclass:: libertem.common.progress.ProgressState
    :members:
