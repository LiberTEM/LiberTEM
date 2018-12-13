import random

from libertem import api
from libertem.io.dataset.base import DataSet, Partition


class ErrorInjectingDataSet(DataSet):
    def __init__(self, base_ds):
        self.base_ds = base_ds

    def get_partitions(self):
        def err_fn():
            if random.random() > 0.9:
                raise Exception("wat")

        for partition in self.base_ds.get_partitions():
            yield ErrorInjectingPartition(
                base_partition=partition,
                err_fn=err_fn,
                dataset=self,
                dtype=partition.dtype,
                partition_slice=partition.slice,
            )

    @property
    def shape(self):
        return self.base_ds.shape

    @property
    def raw_shape(self):
        return self.base_ds.raw_shape

    @property
    def dtype(self):
        return self.base_ds.dtype

    def check_valid(self):
        return self.base_ds.check_valid()

    def get_diagnostics(self):
        return self.base_ds.get_diagnostics()


class ErrorInjectingPartition(Partition):
    def __init__(self, base_partition, err_fn, *args, **kwargs):
        self.base_partition = base_partition
        self.err_fn = err_fn
        super().__init__(*args, **kwargs)

    def get_tiles(self, crop_to=None):
        for tile in self.base_partition.get_tiles():
            yield tile
            self.err_fn()


if __name__ == "__main__":
    with api.Context() as ctx:
        base_ds = ctx.load(
            "blo",
            path="/home/clausen/Data/127.0.0.1.blo",
            tileshape=(1, 8, 144, 144),
        )
        ds = ErrorInjectingDataSet(base_ds)
        analysis = ctx.create_sum_analysis(dataset=ds)
        ctx.run(analysis)
