import sparse
import numpy as np

from libertem.corrections import CorrectionSet
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


class TilingUDF(UDF):
    def process_tile(self, tile):
        pass

    def get_tiling_preferences(self):
        return {
            "depth": UDF.TILE_DEPTH_DEFAULT,
            "total_size": UDF.TILE_SIZE_MAX,
        }


class TilingUDFBestFit(UDF):
    def process_tile(self, tile):
        pass

    def get_tiling_preferences(self):
        return {
            "depth": UDF.TILE_DEPTH_DEFAULT,
            "total_size": UDF.TILE_SIZE_BEST_FIT,
        }


class TilingUDFFrame(UDF):
    def process_frame(self, frame):
        pass


class TilingUDFPartition(UDF):
    def process_partition(self, partition):
        pass


def test_get_scheme_tile(default_raw):
    neg = Negotiator()
    p = next(default_raw.get_partitions())
    udf = TilingUDFBestFit()
    scheme = neg.get_scheme(
        udfs=[udf],
        dataset=default_raw,
        approx_partition_shape=p.shape,
        read_dtype=np.float32,
        roi=None
    )
    assert scheme.shape.sig.dims == 2
    print(neg._get_udf_size_pref(udf))
    print(scheme._debug)
    print(p.shape)
    assert tuple(scheme.shape) == (64, 32, 128)


def test_get_scheme_frame(default_raw):
    neg = Negotiator()
    p = next(default_raw.get_partitions())
    udf = TilingUDFFrame()
    scheme = neg.get_scheme(
        udfs=[udf],
        dataset=default_raw,
        approx_partition_shape=p.shape,
        read_dtype=np.float32,
        roi=None
    )
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (16, 128, 128)


def test_get_scheme_partition(default_raw):
    neg = Negotiator()
    p = next(default_raw.get_partitions())
    udf = TilingUDFPartition()
    scheme = neg.get_scheme(
        udfs=[udf],
        dataset=default_raw,
        approx_partition_shape=p.shape,
        read_dtype=np.float32,
        roi=None
    )
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
    udf = TilingUDFBestFit()
    scheme = neg.get_scheme(
        udfs=[udf],
        dataset=dataset,
        approx_partition_shape=p.shape,
        read_dtype=np.float32,
        roi=None
    )
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (65, 28, 144)


def test_get_scheme_upper_size_roi():
    """
    Confirm that a small ROI will not be split
    up unnecessarily.
    """
    data = _mk_random(size=(1024, 144, 144))
    dataset = MemoryDataSet(
        data=data,
        num_partitions=1,
        sig_dims=2
    )

    roi = np.zeros(dataset.shape.nav, dtype=bool)
    # All in a single partition here
    roi[0] = True
    roi[512] = True
    roi[-1] = True

    neg = Negotiator()
    p = next(dataset.get_partitions())
    udf = TilingUDFBestFit()
    scheme = neg.get_scheme(
        udfs=[udf],
        approx_partition_shape=p.shape,
        dataset=dataset,
        read_dtype=np.float32,
        roi=roi
    )
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (3, 144, 144)


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
    udf = TilingUDFBestFit()
    scheme = neg.get_scheme(
        udfs=[udf],
        approx_partition_shape=p.shape,
        dataset=dataset,
        read_dtype=np.float32,
        roi=None
    )
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
    scheme = neg.get_scheme(
        udfs=[udf],
        approx_partition_shape=p.shape,
        dataset=dataset,
        read_dtype=np.float32,
        roi=None
    )
    print(scheme._debug)
    assert scheme._debug["need_decode"]
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (17, 930, 16)


def test_correction_size_overflow():
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

    excluded_coords = np.array([
        (930, ),
        (16, )
    ])
    excluded_pixels = sparse.COO(coords=excluded_coords, shape=dataset.shape.sig, data=True)
    corr = CorrectionSet(excluded_pixels=excluded_pixels)

    scheme = neg.get_scheme(
        udfs=[udf], approx_partition_shape=p.shape,
        dataset=dataset,
        read_dtype=np.float32, roi=None,
        corrections=corr,
    )
    print(scheme._debug)
    assert scheme._debug["need_decode"]
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (4, 1860, 32)


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
    scheme = neg.get_scheme(
        udfs=[udf],
        approx_partition_shape=p.shape,
        dataset=dataset,
        read_dtype=np.float32,
        roi=None
    )
    print(scheme._debug)
    assert not scheme._debug["need_decode"]
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (32, 1860, 2048)


def test_multi_by_frame_wins():
    by_frame = TilingUDFFrame()
    other_unlimited = UDFUnlimitedDepth()
    other_best_fit = TilingUDFBestFit()
    other_deep = UDFWithLargeDepth()

    udfs = [
        by_frame,
        other_unlimited,
        other_best_fit,
        other_deep,
    ]

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
    scheme = neg.get_scheme(
        udfs=udfs,
        dataset=dataset,
        approx_partition_shape=p.shape,
        read_dtype=np.float32,
        roi=None
    )
    print(scheme._debug)
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (1, 1860, 2048)


def test_multi_no_by_frame_small_size_wins():
    other_unlimited = UDFUnlimitedDepth()
    other_best_fit = TilingUDFBestFit()
    other_deep = UDFWithLargeDepth()

    udfs = [
        other_unlimited,
        other_best_fit,
        other_deep,
    ]

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
    scheme = neg.get_scheme(
        udfs=udfs,
        dataset=dataset,
        approx_partition_shape=p.shape,
        read_dtype=np.float32,
        roi=None
    )
    print(scheme._debug)
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (17, 930, 16)


def test_multi_partition_wins():
    # FIXME: the constellatin partition+tile or partition+frame or similar
    # can be optimized by using a fitting tiling scheme for tile/frame processing
    # and accumulating the whole partition into a buffer, then running process_partition
    # after the tile loop.
    other_unlimited = UDFUnlimitedDepth()
    other_best_fit = TilingUDFBestFit()
    other_deep = UDFWithLargeDepth()
    udf_partition = TilingUDFPartition()

    udfs = [
        udf_partition,
        other_unlimited,
        other_best_fit,
        other_deep,
    ]

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
    scheme = neg.get_scheme(
        udfs=udfs,
        approx_partition_shape=p.shape,
        dataset=dataset,
        read_dtype=np.float32,
        roi=None
    )
    print(scheme._debug)
    assert scheme.shape.sig.dims == 2
    assert tuple(scheme.shape) == (32, 1860, 2048)
