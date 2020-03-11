import warnings

import numpy as np

# For backwards compatibility purposes after move
from libertem.udf.base import Task  # noqa: F401


class Job(object):
    """
    Abstract base class for Job classes.

    Passing an instance of an :code:`Job` sub-class to
    :meth:`libertem.api.Context.run` will generate a :class:`numpy.ndarray`. The
    shape, type and content of this array is governed by the specific
    implementation of the :code:`Job` sub-class.

    .. versionchanged:: 0.3.0
        :code:`Job` is now an abstract base job for documentation purposes
        to hide any implementation details from the user. The previous
        :code:`Job` base class is now called :class:`BaseJob`.

    .. deprecated:: 0.4.0
        Use :ref:`user-defined functions` instead. See also :ref:`job deprecation`
    """
    pass


class BaseJob(Job):
    """
    A computation on a DataSet. Inherit from this class and implement ``get_tasks``
    to yield tasks for your specific computation.

    .. versionadded:: 0.3.0
        Renamed :code:`Job` to :code:`BaseJob`

    .. deprecated:: 0.4.0
        Use :ref:`user-defined functions` instead. See also :ref:`job deprecation`

    """

    def __init__(self, dataset):
        warnings.warn(
            "The Job API is deprecated and will be removed after version 0.6.0. See "
            "https://libertem.github.io/LiberTEM/changelog.html#job-deprecation "
            "for details and a migration guide. "
            "Info: Instantiating %s" % type(self),
            DeprecationWarning, stacklevel=2
        )
        self.dataset = dataset

    def get_tasks(self):
        """
        Yields
        ------
        Task
            ...
        """
        raise NotImplementedError()

    def get_result_shape(self):
        raise NotImplementedError()

    def get_result_dtype(self):
        dtype = np.dtype(self.dataset.dtype)
        if dtype.kind in ('u', 'i'):
            dtype = np.dtype("float32")
        return dtype

    def get_result_buffer(self):
        shape = self.get_result_shape()
        dtype = self.get_result_dtype()
        return np.zeros(shape, dtype=dtype)


class ResultTile(object):
    @property
    def dtype(self):
        raise NotImplementedError

    def reduce_into_result(self, result):
        raise NotImplementedError
