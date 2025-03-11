import os
import sys
import random
from pathlib import Path
import glob

import numpy as np
import pytest
from PIL import Image
import cloudpickle

from libertem.io.dataset.tvips import TVIPSDataSet, get_filenames
from libertem.udf.raw import PickUDF
from libertem.common import Shape
from libertem.udf.sumsigudf import SumSigUDF
from libertem.io.dataset.base import (
    TilingScheme, BufferedBackend, MMapBackend, DirectBackend
)
from libertem.common.buffers import reshaped_view

from utils import dataset_correction_verification, get_testdata_path, ValidationUDF, roi_as_sparse

TVIPS_TESTDATA_PATH = os.path.join(get_testdata_path(), 'TVIPS', 'rec_20200623_080237_000.tvips')
HAVE_TVIPS_TESTDATA = os.path.exists(TVIPS_TESTDATA_PATH)

TVIPS_REFERENCE_DATA_PATTERN = os.path.join(
    get_testdata_path(), 'TVIPS', 'rec_20200623_080237_*.tif'
)

needsdata = pytest.mark.skipif(not HAVE_TVIPS_TESTDATA, reason="need .tvips testdata")  # NOQA


def get_reference_files():
    files = glob.glob(TVIPS_REFERENCE_DATA_PATTERN)
    return list(sorted(files, key=lambda fn: int(fn[-6:-4])))


def get_reference_array():
    files = get_reference_files()
    arrs = [
        np.asarray(Image.open(fn))
        for fn in files
    ]
    return np.stack(arrs)


@pytest.fixture(scope='module')
def default_tvips_raw():
    return get_reference_array()


@needsdata
def test_comparison(default_tvips, default_tvips_raw, lt_ctx_fast):
    udf = ValidationUDF(
        reference=reshaped_view(default_tvips_raw, (-1, *tuple(default_tvips.shape.sig)))
    )
    lt_ctx_fast.run_udf(udf=udf, dataset=default_tvips)


@needsdata
def test_comparison_roi(default_tvips, default_tvips_raw, lt_ctx_fast):
    roi = np.random.choice(
        [True, False],
        size=tuple(default_tvips.shape.nav),
        p=[0.1, 0.9]
    )
    udf = ValidationUDF(reference=default_tvips_raw[roi])
    lt_ctx_fast.run_udf(udf=udf, dataset=default_tvips, roi=roi)


@pytest.fixture
def default_tvips(lt_ctx):
    nav_shape = (10,)
    ds = lt_ctx.load(
        "tvips",
        path=TVIPS_TESTDATA_PATH,
        nav_shape=nav_shape,
        io_backend=MMapBackend(),
        num_partitions=4,
    )
    return ds


@pytest.fixture
def buffered_tvips(lt_ctx):
    nav_shape = (10,)
    ds = lt_ctx.load(
        "tvips",
        path=TVIPS_TESTDATA_PATH,
        nav_shape=nav_shape,
        io_backend=BufferedBackend(),
    )
    return ds


@pytest.fixture
def direct_tvips(lt_ctx):
    nav_shape = (10,)
    ds = lt_ctx.load(
        "tvips",
        path=TVIPS_TESTDATA_PATH,
        nav_shape=nav_shape,
        io_backend=DirectBackend(),
    )
    return ds


def test_glob(tmp_path, naughty_filename):
    naughty_dir = tmp_path / naughty_filename
    naughty_dir.mkdir()

    all_naughty = []
    for k in (0, 1, 2):
        naughty = naughty_dir / (naughty_filename + f'_{k:03}.tvips')
        print(naughty)
        naughty.touch()
        all_naughty.append(naughty)

    filenames_1 = {
        Path(f)
        for f in get_filenames(naughty)
    }
    print(filenames_1)
    target = set(all_naughty)
    assert filenames_1 == target


@needsdata
def test_detect(lt_ctx):
    params = TVIPSDataSet.detect_params(TVIPS_TESTDATA_PATH, lt_ctx.executor)["parameters"]
    assert params == {
        "path": TVIPS_TESTDATA_PATH,
        "nav_shape": (10,),
        "sig_shape": (512, 512),
        "sync_offset": 0,
    }


@needsdata
def test_positive_sync_offset(default_tvips, lt_ctx):
    udf = SumSigUDF()
    sync_offset = 2

    ds_with_offset = TVIPSDataSet(
        path=TVIPS_TESTDATA_PATH, sync_offset=sync_offset,
        num_partitions=2,
    )
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == 2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (3,) + tuple(ds_with_offset.shape.sig),
        sig_dims=ds_with_offset.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds_with_offset.shape,
    )

    tiles = p0.get_tiles(tiling_scheme)
    t0 = next(tiles)
    assert tuple(t0.tile_slice.origin) == (0, 0, 0)
    for _ in tiles:
        pass

    for p in ds_with_offset.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p.slice.origin == (5, 0, 0)
    assert p.slice.shape[0] == 5

    result = lt_ctx.run_udf(dataset=default_tvips, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
        :ds_with_offset._meta.image_count - sync_offset
    ]

    assert np.allclose(result, result_with_offset)


@needsdata
def test_negative_sync_offset(default_tvips, lt_ctx):
    udf = SumSigUDF()
    sync_offset = -2

    ds_with_offset = TVIPSDataSet(
        path=TVIPS_TESTDATA_PATH, sync_offset=sync_offset,
        num_partitions=2,
    )
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == -2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (3,) + tuple(ds_with_offset.shape.sig),
        sig_dims=ds_with_offset.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds_with_offset.shape,
    )

    tiles = p0.get_tiles(tiling_scheme)
    t0 = next(tiles)
    assert tuple(t0.tile_slice.origin) == (2, 0, 0)
    for _ in tiles:
        pass

    for p in ds_with_offset.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p.slice.origin == (5, 0, 0)
    assert p.slice.shape[0] == 5

    result = lt_ctx.run_udf(dataset=default_tvips, udf=udf)
    result = result['intensity'].raw_data[:default_tvips._meta.image_count - abs(sync_offset)]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


@needsdata
def test_offset_smaller_than_nav_shape(lt_ctx):
    nav_shape = (10,)
    sync_offset = -1030

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "tvips",
            path=TVIPS_TESTDATA_PATH,
            nav_shape=nav_shape,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-10, 10\), which is \(-image_count, image_count\)"
        )


@needsdata
def test_offset_greater_than_nav_shape(lt_ctx):
    nav_shape = (10,)
    sync_offset = 1030

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "tvips",
            path=TVIPS_TESTDATA_PATH,
            nav_shape=nav_shape,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-10, 10\), which is \(-image_count, image_count\)"
        )


@needsdata
@pytest.mark.with_numba
def test_read(default_tvips):
    partitions = default_tvips.get_partitions()
    p = next(partitions)
    assert len(p.shape) == 3
    assert tuple(p.shape[1:]) == (512, 512)

    tileshape = Shape(
        (2,) + tuple(default_tvips.shape.sig),
        sig_dims=default_tvips.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_tvips.shape,
    )

    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == (2, 512, 512)
    for _ in tiles:
        pass


@needsdata
def test_scheme_too_large(default_tvips):
    partitions = default_tvips.get_partitions()
    p = next(partitions)
    depth = p.shape[0]

    # we make a tileshape that is too large for the partition here:
    tileshape = Shape(
        (depth + 1,) + tuple(default_tvips.shape.sig),
        sig_dims=default_tvips.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_tvips.shape,
    )

    # tile shape is clamped to partition shape:
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == tuple((depth,) + default_tvips.shape.sig)
    for _ in tiles:
        pass


@needsdata
def test_pickle_is_small(default_tvips):
    pickled = cloudpickle.dumps(default_tvips)
    cloudpickle.loads(pickled)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 2 * 1024


@needsdata
@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_correction(default_tvips, lt_ctx, with_roi):
    ds = default_tvips

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[1] = True
        roi[3] = True
        roi[7] = True
    else:
        roi = None
    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx, exclude=[(45, 144), (124, 30)])
    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


@needsdata
@pytest.mark.with_numba
def test_with_roi(default_tvips, lt_ctx):
    udf = PickUDF()
    roi = np.zeros(default_tvips.shape.nav, dtype=bool)
    roi[0] = 1
    res = lt_ctx.run_udf(udf=udf, dataset=default_tvips, roi=roi)
    np.array(res['intensity']).shape == (1, 512, 512)


@needsdata
def test_diagnostics(default_tvips):
    print(default_tvips.diagnostics)


@needsdata
@pytest.mark.dist
def test_tvips_dist(dist_ctx):
    nav_shape = (10,)
    ds = TVIPSDataSet(path=TVIPS_TESTDATA_PATH, nav_shape=nav_shape)
    ds = ds.initialize(dist_ctx.executor)
    analysis = dist_ctx.create_sum_analysis(dataset=ds)
    results = dist_ctx.run(analysis)
    assert results[0].raw_data.shape == (512, 512)


@needsdata
def test_reshape_nav(lt_ctx):
    udf = SumSigUDF()

    ds_with_1d_nav = lt_ctx.load("tvips", path=TVIPS_TESTDATA_PATH, nav_shape=(8,))
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    ds_with_2d_nav = lt_ctx.load("tvips", path=TVIPS_TESTDATA_PATH, nav_shape=(4, 2))
    result_with_2d_nav = lt_ctx.run_udf(dataset=ds_with_2d_nav, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load("tvips", path=TVIPS_TESTDATA_PATH, nav_shape=(2, 2, 2))
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


@needsdata
def test_incorrect_sig_shape(lt_ctx):
    nav_shape = (10,)
    sig_shape = (5, 5)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "tvips",
            path=TVIPS_TESTDATA_PATH,
            nav_shape=nav_shape,
            sig_shape=sig_shape
        )
    assert e.match(
        r"sig_shape must be of size: 262144"
    )


@needsdata
def test_compare_backends(lt_ctx, default_tvips, buffered_tvips):
    x = random.choice(range(default_tvips.shape.nav[0]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_tvips,
        x=x,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=buffered_tvips,
        x=x,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


@needsdata
@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="No support for direct I/O on Mac OS X"
)
def test_compare_direct_to_mmap(lt_ctx, default_tvips, direct_tvips):
    x = random.choice(range(default_tvips.shape.nav[0]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_tvips,
        x=x,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=direct_tvips,
        x=x,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


@needsdata
@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_compare_backends_sparse(lt_ctx, default_tvips, buffered_tvips, as_sparse):
    roi = np.zeros(default_tvips.shape.nav, dtype=bool).reshape((-1,))
    roi[0] = True
    roi[1] = True
    roi[6] = True
    roi[-1] = True
    if as_sparse:
        roi = roi_as_sparse(roi)
    mm_f0 = lt_ctx.run_udf(dataset=default_tvips, udf=PickUDF(), roi=roi)['intensity']
    buffered_f0 = lt_ctx.run_udf(dataset=buffered_tvips, udf=PickUDF(), roi=roi)['intensity']

    assert np.allclose(mm_f0, buffered_f0)


def test_bad_params(ds_params_tester, standard_bad_ds_params):
    args = ("tvips", TVIPS_TESTDATA_PATH)
    for params in standard_bad_ds_params:
        ds_params_tester(*args, **params)


@needsdata
def test_no_num_partitions(lt_ctx):
    ds = lt_ctx.load(
        "tvips",
        path=TVIPS_TESTDATA_PATH,
    )
    lt_ctx.run_udf(dataset=ds, udf=SumSigUDF())
