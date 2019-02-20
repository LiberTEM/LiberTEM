import numpy as np


class Job(object):
    """
    A computation on a DataSet. Inherit from this class and implement ``get_tasks``
    to yield tasks for your specific computation.
    """

    def __init__(self, dataset):
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


class Task(object):
    """
    A computation on a partition. Inherit from this class and implement ``__call__``
    for your specific computation.
    """

    def __init__(self, partition, idx):
        self.partition = partition
        self.idx = idx

    def get_locations(self):
        return self.partition.get_locations()

    def __call__(self):
        raise NotImplementedError()


class ResultTile(object):
    @property
    def dtype(self):
        raise NotImplementedError

    def reduce_into_result(self, result):
        raise NotImplementedError
