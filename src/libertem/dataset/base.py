class DataSet(object):
    def get_partitions(self):
        raise NotImplementedError()


class Partition(object):
    def __init__(self, dataset, dtype, partition_slice):
        self.dataset = dataset
        self.dtype = dtype
        self.slice = partition_slice

    def get_tiles(self):
        raise NotImplementedError()

    def get_locations(self):
        raise NotImplementedError()
