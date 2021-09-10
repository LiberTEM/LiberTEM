import os
import importlib

import numpy as np

from libertem.udf import UDF
from libertem.io.dataset.memory import MemoryDataSet

# A bit of gymnastics to import the test utilities from outside the package
basedir = os.path.dirname(__file__)
location = os.path.join(basedir, "../../tests/utils.py")
spec = importlib.util.spec_from_file_location("utils", location)
utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(utils)


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
    data = utils._mk_random(size=(64, 64, 64, 64), dtype="float32")
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
    data = utils._mk_random(size=(64, 64, 64, 64), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(4, 64, 64),
                            num_partitions=2, sig_dims=2)

    test = Test_UDF_w_set_coords()
    benchmark(lt_ctx.run_udf, udf=test, dataset=dataset)


class Test_UDF_frame_coords(UDF):
    def get_result_buffers(self):
        return {
            'test': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_frame(self, frame):
        assert self.meta.coordinates is not None


def test_get_tiles_by_frame_w_coords_roi(lt_ctx, benchmark):
    data = utils._mk_random(size=(64, 64, 64, 64), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(4, 64, 64),
                            num_partitions=2, sig_dims=2)
    roi = np.random.choice([True, False], dataset.shape.nav, p=[0.9, 0.1])
    test = Test_UDF_frame_coords()
    benchmark(lt_ctx.run_udf, udf=test, dataset=dataset, roi=roi)
