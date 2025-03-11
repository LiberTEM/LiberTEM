import os

import numpy as np
import pytest

from libertem.io.dataset.mrc import MRCDataSet
from libertem.udf.sumsigudf import SumSigUDF
from libertem.io.dataset.base import BufferedBackend, TilingScheme
from libertem.common import Shape
from libertem.common.buffers import reshaped_view
from libertem.udf.raw import PickUDF

from utils import dataset_correction_verification, get_testdata_path, ValidationUDF, roi_as_sparse

try:
    import mrcfile
except ModuleNotFoundError:
    mrcfile = None


MRC_TESTDATA_PATH = os.path.join(
    get_testdata_path(), 'mrc', '20200821_92978_movie.mrc',
)
HAVE_MRC_TESTDATA = os.path.exists(MRC_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_MRC_TESTDATA, reason="need .mrc testdata")  # NOQA


@pytest.fixture
def default_mrc(lt_ctx):
    ds = lt_ctx.load("mrc", path=MRC_TESTDATA_PATH)
    return ds


@pytest.fixture
def buffered_mrc(lt_ctx):
    # FIXME: not yet supported
    ds = lt_ctx.load("mrc", path=MRC_TESTDATA_PATH, io_backend=BufferedBackend())
    return ds


@pytest.fixture(scope='module')
def default_mrc_raw():
    mrc = mrcfile.open(MRC_TESTDATA_PATH)
    return mrc.data


def test_simple_open(default_mrc):
    assert tuple(default_mrc.shape) == (4, 1024, 1024)


def test_check_valid(default_mrc):
    default_mrc.check_valid()


@pytest.mark.skipif(mrcfile is None, reason="No mrcfile found")
def test_comparison(default_mrc, default_mrc_raw, lt_ctx_fast):
    udf = ValidationUDF(
        reference=reshaped_view(default_mrc_raw, (-1, *tuple(default_mrc.shape.sig)))
    )
    lt_ctx_fast.run_udf(udf=udf, dataset=default_mrc)


@pytest.mark.skipif(mrcfile is None, reason="No mrcfile found")
@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_comparison_roi(default_mrc, default_mrc_raw, lt_ctx_fast, as_sparse):
    roi = np.random.choice(
        [True, False],
        size=tuple(default_mrc.shape.nav),
        p=[0.5, 0.5]
    )
    ref_data = default_mrc_raw[roi]
    if as_sparse:
        roi = roi_as_sparse(roi)
    udf = ValidationUDF(reference=ref_data)
    lt_ctx_fast.run_udf(udf=udf, dataset=default_mrc, roi=roi)


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
def test_correction(default_mrc, lt_ctx, with_roi):
    ds = default_mrc

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


def test_detect_1(lt_ctx):
    fpath = MRC_TESTDATA_PATH
    assert MRCDataSet.detect_params(
        path=fpath,
        executor=lt_ctx.executor,
    )["parameters"] == {
        'path': fpath,
        'nav_shape': (2, 2),
        'sig_shape': (1024, 1024)
    }


def test_detect_2(lt_ctx):
    assert MRCDataSet.detect_params(
        path="nofile.someext",
        executor=lt_ctx.executor,
    ) is False


@pytest.mark.dist
def test_mrc_dist(dist_ctx):
    ds = MRCDataSet(path=MRC_TESTDATA_PATH)
    ds = ds.initialize(dist_ctx.executor)
    analysis = dist_ctx.create_sum_analysis(dataset=ds)
    roi = np.random.choice([True, False], size=4)
    results = dist_ctx.run(analysis, roi=roi)
    assert results[0].raw_data.shape == (1024, 1024)


def test_positive_sync_offset(lt_ctx, default_mrc):
    # nav shape 4, we go over the trailing edge
    udf = PickUDF()
    sync_offset = 2

    roi = np.zeros(default_mrc.shape.nav, dtype=bool)
    flat_roi = reshaped_view(roi, -1)
    flat_roi[2:4] = True

    ref = lt_ctx.run_udf(dataset=default_mrc, udf=udf, roi=roi)

    ds_with_offset = lt_ctx.load(
        "mrc", path=MRC_TESTDATA_PATH, nav_shape=(2, 2), sync_offset=sync_offset
    )

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    shape = lt_ctx.run_udf(dataset=ds_with_offset, udf=SumSigUDF())

    print(result_with_offset['intensity'].raw_data.shape)
    assert shape['intensity'].data.shape == (2, 2)
    assert np.allclose(
        result_with_offset['intensity'].raw_data[:2],
        ref['intensity'].raw_data
    )


def test_negative_sync_offset(default_mrc, lt_ctx):
    # nav shape 4
    udf = PickUDF()
    sync_offset = -2

    roi = np.zeros(default_mrc.shape.nav, dtype=bool)
    flat_roi = reshaped_view(roi, -1)
    flat_roi[:2] = True

    ref = lt_ctx.run_udf(dataset=default_mrc, udf=udf, roi=roi)

    ds_with_offset = lt_ctx.load(
        "mrc", path=MRC_TESTDATA_PATH, nav_shape=(2, 2), sync_offset=sync_offset
    )

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    shape = lt_ctx.run_udf(dataset=ds_with_offset, udf=SumSigUDF())

    print(result_with_offset['intensity'].raw_data.shape)
    assert shape['intensity'].data.shape == (2, 2)
    assert np.allclose(
        result_with_offset['intensity'].raw_data[2:],
        ref['intensity'].raw_data
    )


def test_offset_smaller_than_image_count(lt_ctx):
    sync_offset = -20

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "mrc",
            path=MRC_TESTDATA_PATH,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-4, 4\), which is \(-image_count, image_count\)"
    )


def test_offset_greater_than_image_count(lt_ctx):
    sync_offset = 20

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "mrc",
            path=MRC_TESTDATA_PATH,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-4, 4\), which is \(-image_count, image_count\)"
    )


def test_reshape_nav(lt_ctx):
    udf = SumSigUDF()

    ds_with_1d_nav = lt_ctx.load("mrc", path=MRC_TESTDATA_PATH, nav_shape=(4,))
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    ds_with_2d_nav = lt_ctx.load("mrc",  path=MRC_TESTDATA_PATH, nav_shape=(2, 2,))
    result_with_2d_nav = lt_ctx.run_udf(dataset=ds_with_2d_nav, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load("mrc",  path=MRC_TESTDATA_PATH, nav_shape=(1, 2, 2))
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


def test_incorrect_sig_shape(lt_ctx):
    sig_shape = (5, 5)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "mrc",
            path=MRC_TESTDATA_PATH,
            sig_shape=sig_shape
        )
    assert e.match(
        r"sig_shape must be of size: 1048576"
    )


def test_scheme_too_large(default_mrc):
    partitions = default_mrc.get_partitions()
    p = next(partitions)
    depth = p.shape[0]

    # we make a tileshape that is too large for the partition here:
    tileshape = Shape(
        (depth + 1,) + tuple(default_mrc.shape.sig),
        sig_dims=default_mrc.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_mrc.shape,
    )

    # tile shape is clamped to partition shape:
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == tuple((depth,) + default_mrc.shape.sig)


def test_diagnostics(default_mrc):
    assert {"name": "dtype", "value": "float32"} in default_mrc.get_diagnostics()


def test_bad_params(ds_params_tester, standard_bad_ds_params):
    args = ("mrc", MRC_TESTDATA_PATH)
    for params in standard_bad_ds_params:
        ds_params_tester(*args, **params)


def test_num_partitions(lt_ctx):
    ds = lt_ctx.load(
        "mrc",
        path=MRC_TESTDATA_PATH,
        num_partitions=3,
    )
    assert len(list(ds.get_partitions())) == 3
