from libertem.udf import UDF
from libertem.io.dataset.memory import MemoryDataSet
from tests.utils import _mk_random


class Test_UDF(UDF):
    def get_result_buffers(self):
        return {
            'test': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_partition(self, partition):
        pass


def test_get_tiles_by_partition(lt_ctx, benchmark):
    data = _mk_random(size=(64, 64, 64, 64), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(4, 64, 64),
                            num_partitions=2, sig_dims=2)

    test = Test_UDF()
    benchmark(lt_ctx.run_udf, udf=test, dataset=dataset)


class Test_UDF_w_set_coords(UDF):
    def get_result_buffers(self):
        return {
            'test': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_partition(self, partition):
        assert self.meta.coordinates is not None


def test_get_tiles_by_partition_w_coords(lt_ctx, benchmark):
    data = _mk_random(size=(64, 64, 64, 64), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(4, 64, 64),
                            num_partitions=2, sig_dims=2)

    test = Test_UDF_w_set_coords()
    benchmark(lt_ctx.run_udf, udf=test, dataset=dataset)
