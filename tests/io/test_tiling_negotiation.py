import numpy as np

from libertem.io.dataset.base import Negotiator
from libertem.io.dataset.memory import MemoryDataSet
from libertem.udf import UDF

from utils import _mk_random


def test_scale_factors():
    neg = Negotiator()
    # scenario: k2is.
    assert neg._get_scale_factors(
        shape=(930, 16),
        containing_shape=(1860, 2048),
        size=1024,
    ) == [1, 1]

    assert neg._get_scale_factors(
        shape=(930, 16),
        containing_shape=(1860, 2048),
        size=930 * 16 * 16
    ) == [2, 8]

    assert neg._get_scale_factors(
        shape=(930, 16),
        containing_shape=(1860, 2048),
        size=930 * 16 * 128
    ) == [2, 64]

    # slightly above, but not enough to fit another block: we err on the small side
    assert neg._get_scale_factors(
        shape=(930, 16),
        containing_shape=(1860, 2048),
        size=930 * 16 * 129
    ) == [2, 64]

    # larger size than we can accomodate with our containing shape:
    assert neg._get_scale_factors(
        shape=(930, 16),
        containing_shape=(1860, 2048),
        size=1860 * 2048 * 2
    ) == [2, 128]


class TestUDF(UDF):
    def process_tile(self, tile):
        pass

    def get_tiling_preferences(self):
        return {
            "depth": UDF.TILE_DEPTH_DEFAULT,
            "total_size": UDF.TILE_SIZE_MAX,
        }


class TestUDFBestFit(UDF):
    def process_tile(self, tile):
        pass

    def get_tiling_preferences(self):
        return {
            "depth": UDF.TILE_DEPTH_DEFAULT,
            "total_size": UDF.TILE_SIZE_BEST_FIT,
        }


class TestUDFFrame(UDF):
    def process_frame(self, frame):
        pass


class TestUDFPartition(UDF):
    def process_partition(self, partition):
        pass


def test_get_scheme_tile(default_raw):
    neg = Negotiator()
    p = next(default_raw.get_partitions())
    udf = TestUDFBestFit()
    scheme = neg.get_scheme(udf=udf, partition=p, read_dtype=np.float32, roi=None)
    assert scheme.shape.sig.dims == 2
    print(neg._get_udf_size_pref(udf))
    print(scheme._debug)
    print(p.shape)
    assert tuple(scheme.shape) == (64, 32, 128)


def test_get_scheme_frame(default_raw):
    neg = Negotiator()
    p = next(default_raw.get_partitions())
    udf = TestUDFFrame()
    scheme = neg.get_scheme(udf=udf, partition=p, read_dtype=np.float32, roi=None)
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (16, 128, 128)


def test_get_scheme_partition(default_raw):
    neg = Negotiator()
    p = next(default_raw.get_partitions())
    udf = TestUDFPartition()
    scheme = neg.get_scheme(udf=udf, partition=p, read_dtype=np.float32, roi=None)
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (128, 128, 128)


def test_get_scheme_upper_size_1():
    """
    Test that will hit the 2**20 default size
    """
    data = _mk_random(size=(1024, 144, 144))
    dataset = MemoryDataSet(
        data=data,
        num_partitions=1,
        sig_dims=2
    )

    neg = Negotiator()
    p = next(dataset.get_partitions())
    udf = TestUDFBestFit()
    scheme = neg.get_scheme(udf=udf, partition=p, read_dtype=np.float32, roi=None)
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (65, 28, 144)


def test_get_scheme_upper_size_2():
    """
    Test that will hit the 2**20 default size
    """
    data = _mk_random(size=(2048, 264, 264))
    dataset = MemoryDataSet(
        data=data,
        num_partitions=1,
        sig_dims=2,
        base_shape=(1, 8, 264),
    )

    neg = Negotiator()
    p = next(dataset.get_partitions())
    udf = TestUDFBestFit()
    scheme = neg.get_scheme(udf=udf, partition=p, read_dtype=np.float32, roi=None)
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (124, 8, 264)


class UDFWithLargeDepth(UDF):
    def process_tile(self, tile):
        pass

    def get_tiling_preferences(self):
        return {
            "depth": 128,
            "total_size": UDF.TILE_SIZE_BEST_FIT,
        }


class UDFUnlimitedDepth(UDF):
    def process_tile(self, tile):
        pass

    def get_tiling_preferences(self):
        return {
            "depth": UDF.TILE_DEPTH_MAX,
            "total_size": UDF.TILE_SIZE_MAX,
        }


def test_limited_depth():
    data = _mk_random(size=(32, 1860, 2048))
    dataset = MemoryDataSet(
        data=data,
        num_partitions=1,
        sig_dims=2,
        base_shape=(1, 930, 16),
        force_need_decode=True,
    )

    neg = Negotiator()
    p = next(dataset.get_partitions())
    udf = UDFWithLargeDepth()
    scheme = neg.get_scheme(udf=udf, partition=p, read_dtype=np.float32, roi=None)
    print(scheme._debug)
    assert scheme._debug["need_decode"]
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (17, 930, 16)


def test_depth_max_size_max():
    data = _mk_random(size=(32, 1860, 2048))
    dataset = MemoryDataSet(
        data=data,
        num_partitions=1,
        sig_dims=2,
        base_shape=(1, 930, 16),
        force_need_decode=False,
    )

    neg = Negotiator()
    p = next(dataset.get_partitions())
    udf = UDFUnlimitedDepth()
    scheme = neg.get_scheme(udf=udf, partition=p, read_dtype=np.float32, roi=None)
    print(scheme._debug)
    assert not scheme._debug["need_decode"]
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (32, 1860, 2048)
