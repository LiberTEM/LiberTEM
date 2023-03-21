import numpy as np

from libertem.udf import UDF
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


class SimpleTestByPartitionUDF(UDF):
    def get_result_buffers(self):
        return {
            'test': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_partition(self, partition):
        # coordinates of frames in two partitions of 8x8x8x8 data
        part_1_coords = [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [0, 5], [0, 6], [0, 7],
                         [1, 0], [1, 1], [1, 2], [1, 3], [1, 4], [1, 5], [1, 6], [1, 7],
                         [2, 0], [2, 1], [2, 2], [2, 3], [2, 4], [2, 5], [2, 6], [2, 7],
                         [3, 0], [3, 1], [3, 2], [3, 3], [3, 4], [3, 5], [3, 6], [3, 7]]
        part_2_coords = [[4, 0], [4, 1], [4, 2], [4, 3], [4, 4], [4, 5], [4, 6], [4, 7],
                         [5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [5, 5], [5, 6], [5, 7],
                         [6, 0], [6, 1], [6, 2], [6, 3], [6, 4], [6, 5], [6, 6], [6, 7],
                         [7, 0], [7, 1], [7, 2], [7, 3], [7, 4], [7, 5], [7, 6], [7, 7]]
        if self.meta.slice.origin == (0, 0, 0):
            assert np.allclose(self.meta.coordinates, part_1_coords)
        elif self.meta.slice.origin == (32, 0, 0):
            assert np.allclose(self.meta.coordinates, part_2_coords)


def test_tiles_by_partition(lt_ctx):
    data = _mk_random(size=(8, 8, 8, 8), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(32, 8, 8),
                            num_partitions=2, sig_dims=2)

    test = SimpleTestByPartitionUDF()
    lt_ctx.run_udf(dataset=dataset, udf=test)


class SimpleTestByFrameUDF(UDF):
    def get_result_buffers(self):
        return {
            'test': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_frame(self, frame):
        # only two slices, each with one frame
        if self.meta.slice.origin == (0, 0, 0):
            assert np.allclose(self.meta.coordinates, [[0, 0]])
        elif self.meta.slice.origin == (1, 0, 0):
            assert np.allclose(self.meta.coordinates, [[0, 1]])


def test_tiles_by_frame(lt_ctx):
    data = _mk_random(size=(8, 8, 8, 8), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(4, 8, 8), sync_offset=62,
                            num_partitions=2, sig_dims=2)

    test = SimpleTestByFrameUDF()
    lt_ctx.run_udf(dataset=dataset, udf=test)


class SimpleTestByTileZeroSyncOffsetUDF(UDF):
    def get_result_buffers(self):
        return {
            'test': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_tile(self, tile):
        # verifying coordinates of frames of some tile_slices
        if self.meta.slice.origin == (0, 0, 0):
            expected_coords = [[0, 0], [0, 1], [0, 2], [0, 3]]
            assert np.allclose(self.meta.coordinates, expected_coords)
        elif self.meta.slice.origin == (32, 0, 0):
            expected_coords = [[4, 0], [4, 1], [4, 2], [4, 3]]
            assert np.allclose(self.meta.coordinates, expected_coords)
        elif self.meta.slice.origin == (60, 0, 0):
            expected_coords = [[7, 4], [7, 5], [7, 6], [7, 7]]
            assert np.allclose(self.meta.coordinates, expected_coords)


def test_tiles_no_offset(lt_ctx):
    data = _mk_random(size=(8, 8, 8, 8), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(4, 8, 8),
                            num_partitions=2, sig_dims=2)

    test = SimpleTestByTileZeroSyncOffsetUDF()
    lt_ctx.run_udf(dataset=dataset, udf=test)


class SimpleTestByTilePositiveSyncOffsetUDF(UDF):
    def get_result_buffers(self):
        return {
            'test': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_tile(self, tile):
        # only two frames in the tile, which are the last two frames of source data
        expected_coords = [[0, 0], [0, 1]]
        assert np.allclose(self.meta.coordinates, expected_coords)


def test_tiles_positive_offset(lt_ctx):
    data = _mk_random(size=(8, 8, 8, 8), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(4, 8, 8),
                            num_partitions=2, sig_dims=2, sync_offset=62)

    test = SimpleTestByTilePositiveSyncOffsetUDF()
    lt_ctx.run_udf(dataset=dataset, udf=test)


class SimpleTestByTileNegativeSyncOffsetUDF(UDF):
    def get_result_buffers(self):
        return {
            'test': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_tile(self, tile):
        # four frames in the tile, which are the first four frames of source data
        expected_coords = [[7, 6], [7, 7]]
        assert np.allclose(self.meta.coordinates, expected_coords)


def test_tiles_negative_offset(lt_ctx):
    data = _mk_random(size=(8, 8, 8, 8), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(4, 8, 8),
                            num_partitions=2, sig_dims=2, sync_offset=-62)

    test = SimpleTestByTileNegativeSyncOffsetUDF()
    lt_ctx.run_udf(dataset=dataset, udf=test)


class SimpleTestByTileWithROIUDF(UDF):
    def get_result_buffers(self):
        return {
            'test': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_tile(self, tile):
        if self.meta.slice.origin == (0, 0, 0):
            expected_coords = [[0, 0], [0, 7]]
            print("meta:", self.meta.coordinates, "expected:", expected_coords)
            assert np.allclose(self.meta.coordinates, expected_coords)
        elif self.meta.slice.origin == (2, 0, 0):
            expected_coords = [[7, 0], [7, 7]]
            print("meta:", self.meta.coordinates, "expected:", expected_coords)
            assert np.allclose(self.meta.coordinates, expected_coords)


def test_tiles_with_roi(lt_ctx):
    data = _mk_random(size=(8, 8, 8, 8), dtype="float32")
    roi = np.random.choice([False], (8, 8))
    roi[0, 0] = True
    roi[0, 7] = True
    roi[7, 0] = True
    roi[7, 7] = True
    dataset = MemoryDataSet(data=data, tileshape=(4, 8, 8),
                            num_partitions=2, sig_dims=2)

    test = SimpleTestByTileWithROIUDF()
    lt_ctx.run_udf(dataset=dataset, udf=test, roi=roi)


class CoordsInGetTaskData(UDF):
    def get_result_buffers(self):
        return {
            'counter': self.buffer(kind='single', dtype=np.int64),
        }

    def process_tile(self, tile):
        pass

    def get_task_data(self):
        assert self.meta._slice is not None
        assert self.meta.coordinates is not None
        ps = np.zeros((self.meta.coordinates.shape[0],), dtype=bool)
        return {
            'ps': ps,
        }

    def postprocess(self):
        self.results.counter[0] += self.task_data.ps.shape[0]

    def merge(self, dest, src):
        dest.counter += src.counter


def test_coordinates_available(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = lt_ctx.load(
        "memory",
        data=data,
        tileshape=(7, 16, 16),
        num_partitions=2,
        sig_dims=2
    )
    roi = np.random.choice([True, False], dataset.shape.nav)

    udf = CoordsInGetTaskData()
    res = lt_ctx.run_udf(dataset=dataset, udf=udf, roi=roi)
    assert res['counter'].data[0] == np.count_nonzero(roi)
