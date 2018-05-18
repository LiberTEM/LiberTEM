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


class Task(object):
    """
    A computation on a partition. Inherit from this class and implement ``__call__``
    for your specific computation.
    """

    def __init__(self, partition):
        self.partition = partition

    def get_locations(self):
        return self.partition.get_locations()

    def __call__(self):
        raise NotImplementedError()
