from libertem.io.dataset.raw import RawFileDataSet
from libertem.common import Shape
from libertem.io.dataset.base import TilingScheme


def get_first_tile(p0, tiling_scheme):
    return next(p0.get_tiles(tiling_scheme))


def test_positive_sync_offset_raw(lt_ctx, benchmark, raw_data_8x8x8x8_path):
    ds = RawFileDataSet(
        path=raw_data_8x8x8x8_path,
        nav_shape=(8, 8),
        sig_shape=(8, 8),
        dtype="float32",
        enable_direct=False,
        sync_offset=2
    )
    ds.set_num_cores(4)
    ds = ds.initialize(lt_ctx.executor)

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    p0 = next(ds.get_partitions())
    benchmark(get_first_tile, p0=p0, tiling_scheme=tiling_scheme)


def test_negative_sync_offset_raw(lt_ctx, benchmark, raw_data_8x8x8x8_path):
    ds = RawFileDataSet(
        path=raw_data_8x8x8x8_path,
        nav_shape=(8, 8),
        sig_shape=(8, 8),
        dtype="float32",
        enable_direct=False,
        sync_offset=-2
    )
    ds.set_num_cores(4)
    ds = ds.initialize(lt_ctx.executor)

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    p0 = next(ds.get_partitions())
    benchmark(get_first_tile, p0=p0, tiling_scheme=tiling_scheme)


def test_no_sync_offset_raw(lt_ctx, benchmark, raw_data_8x8x8x8_path):
    ds = RawFileDataSet(
        path=raw_data_8x8x8x8_path,
        nav_shape=(8, 8),
        sig_shape=(8, 8),
        dtype="float32",
        enable_direct=False,
    )
    ds.set_num_cores(4)
    ds = ds.initialize(lt_ctx.executor)

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    p0 = next(ds.get_partitions())
    benchmark(get_first_tile, p0=p0, tiling_scheme=tiling_scheme)
