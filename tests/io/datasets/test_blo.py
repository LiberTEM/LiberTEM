import os
import sys
import json
import random

import numpy as np
import pytest
import cloudpickle

from libertem.analysis.raw import PickFrameAnalysis
from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.blo import BloDataSet
from libertem.io.dataset.base import TilingScheme, BufferedBackend, MMapBackend, DirectBackend
from libertem.common import Shape
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.raw import PickUDF

from utils import dataset_correction_verification, get_testdata_path, ValidationUDF, roi_as_sparse

try:
    import rsciio
    import rsciio.blockfile
except ModuleNotFoundError:
    rsciio = None


BLO_TESTDATA_PATH = os.path.join(get_testdata_path(), 'default.blo')
HAVE_BLO_TESTDATA = os.path.exists(BLO_TESTDATA_PATH)
BLO_16BIT_TESTDATA_PATH = os.path.join(get_testdata_path(), 'Stingray16bit_7_11.blo')
HAVE_BLO_16BIT_TESTDATA = os.path.exists(BLO_16BIT_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_BLO_TESTDATA, reason="need .blo testdata")  # NOQA


@pytest.fixture()
def default_blo(lt_ctx):
    ds = lt_ctx.load(
        "blo",
        path=str(BLO_TESTDATA_PATH),
        io_backend=MMapBackend(),
    )
    return ds


@pytest.fixture()
def blo_16bit(lt_ctx):
    ds = lt_ctx.load(
        "blo",
        path=str(BLO_16BIT_TESTDATA_PATH),
        io_backend=MMapBackend(),
    )
    return ds


@pytest.fixture
def buffered_blo(lt_ctx):
    buffered = BufferedBackend()
    ds_buffered = lt_ctx.load(
        "blo",
        path=str(BLO_TESTDATA_PATH),
        io_backend=buffered,
    )
    return ds_buffered


@pytest.fixture
def direct_blo(lt_ctx):
    direct = DirectBackend()
    ds_buffered = lt_ctx.load(
        "blo",
        path=str(BLO_TESTDATA_PATH),
        io_backend=direct,
    )
    return ds_buffered


@pytest.fixture(scope='module')
def default_blo_raw():
    res = rsciio.blockfile.file_reader(str(BLO_TESTDATA_PATH), lazy=True)
    return res[0]['data']


def test_simple_open(default_blo: BloDataSet):
    assert tuple(default_blo.shape) == (90, 121, 144, 144)
    assert default_blo.meta.raw_dtype == np.uint8


def test_check_valid(default_blo):
    assert default_blo.check_valid()


def test_detect():
    assert BloDataSet.detect_params(
        path=str(BLO_TESTDATA_PATH),
        executor=InlineJobExecutor()
    )["parameters"]


def test_read(default_blo):
    partitions = default_blo.get_partitions()
    p = next(partitions)
    # FIXME: partition shape can vary by number of cores
    # assert tuple(p.shape) == (11, 121, 144, 144)
    tileshape = Shape(
        (8,) + tuple(default_blo.shape.sig),
        sig_dims=default_blo.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_blo.shape,
    )

    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == (8, 144, 144)
    for _ in tiles:
        pass


def test_scheme_too_large(default_blo):
    partitions = default_blo.get_partitions()
    p = next(partitions)
    depth = p.shape[0]

    # we make a tileshape that is too large for the partition here:
    tileshape = Shape(
        (depth + 1,) + tuple(default_blo.shape.sig),
        sig_dims=default_blo.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_blo.shape,
    )

    # tile shape is clamped to partition shape:
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == (depth, 144, 144)
    for _ in tiles:
        pass


@pytest.mark.skipif(rsciio is None, reason="No rosettasciio found")
def test_comparison(default_blo, default_blo_raw, lt_ctx_fast):
    reference = default_blo_raw.reshape(-1, *tuple(default_blo.shape.sig))
    udf = ValidationUDF(
        reference=reference,
    )
    lt_ctx_fast.run_udf(udf=udf, dataset=default_blo)


@pytest.mark.skipif(rsciio is None, reason="No rosettasciio found")
def test_comparison_roi(default_blo, default_blo_raw, lt_ctx_fast):
    reference = default_blo_raw.reshape(-1, *tuple(default_blo.shape.sig))
    roi = np.random.choice(
        [True, False],
        size=reference.shape[0],
        p=[0.5, 0.5]
    )
    udf = ValidationUDF(reference=reference[roi])
    lt_ctx_fast.run_udf(udf=udf, dataset=default_blo, roi=roi)


def test_pickle_meta_is_small(default_blo):
    pickled = cloudpickle.dumps(default_blo._meta)
    cloudpickle.loads(pickled)
    assert len(pickled) < 512


def test_pickle_fileset_is_small(default_blo):
    pickled = cloudpickle.dumps(default_blo._get_fileset())
    cloudpickle.loads(pickled)
    assert len(pickled) < 1024


def test_apply_mask_analysis(default_blo, lt_ctx):
    mask = np.ones((144, 144))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_blo)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (90, 121)


def test_sum_analysis(default_blo, lt_ctx):
    analysis = lt_ctx.create_sum_analysis(dataset=default_blo)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (144, 144)


def test_pick_analysis(default_blo, lt_ctx):
    analysis = PickFrameAnalysis(dataset=default_blo, parameters={"x": 16, "y": 16})
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (144, 144)


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
@pytest.mark.slow
def test_correction(default_blo, lt_ctx, with_roi):
    ds = default_blo

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx, exclude=[(55, 92), (61, 31)])
    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


def test_cache_key_json_serializable(default_blo):
    json.dumps(default_blo.get_cache_key())


@pytest.mark.dist
def test_blo_dist(dist_ctx):
    ds = BloDataSet(path=BLO_TESTDATA_PATH)
    ds = ds.initialize(dist_ctx.executor)
    analysis = dist_ctx.create_sum_analysis(dataset=ds)
    results = dist_ctx.run(analysis)
    assert results[0].raw_data.shape == (144, 144)


def test_positive_sync_offset(lt_ctx):
    udf = SumSigUDF()
    sync_offset = 2

    ds = lt_ctx.load(
        "blo", path=BLO_TESTDATA_PATH, nav_shape=(4, 2),
    )

    result = lt_ctx.run_udf(dataset=ds, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    ds_with_offset = lt_ctx.load(
        "blo", path=BLO_TESTDATA_PATH, nav_shape=(4, 2), sync_offset=sync_offset
    )

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
        :ds_with_offset._meta.shape.nav.size - sync_offset
    ]

    assert np.allclose(result, result_with_offset)


def test_negative_sync_offset(lt_ctx):
    udf = SumSigUDF()
    sync_offset = -2

    ds = lt_ctx.load(
        "blo", path=BLO_TESTDATA_PATH, nav_shape=(4, 2),
    )

    result = lt_ctx.run_udf(dataset=ds, udf=udf)
    result = result['intensity'].raw_data[:ds._meta.shape.nav.size - abs(sync_offset)]

    ds_with_offset = lt_ctx.load(
        "blo", path=BLO_TESTDATA_PATH, nav_shape=(4, 2), sync_offset=sync_offset
    )

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


def test_offset_smaller_than_image_count(lt_ctx):
    sync_offset = -10900

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "blo",
            path=BLO_TESTDATA_PATH,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-10890, 10890\), which is \(-image_count, image_count\)"
    )


def test_offset_greater_than_image_count(lt_ctx):
    sync_offset = 10900

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "blo",
            path=BLO_TESTDATA_PATH,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-10890, 10890\), which is \(-image_count, image_count\)"
    )


def test_reshape_nav(lt_ctx):
    udf = SumSigUDF()

    ds_with_1d_nav = lt_ctx.load("blo", path=BLO_TESTDATA_PATH, nav_shape=(8,))
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    ds_with_2d_nav = lt_ctx.load("blo", path=BLO_TESTDATA_PATH, nav_shape=(4, 2))
    result_with_2d_nav = lt_ctx.run_udf(dataset=ds_with_2d_nav, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load("blo", path=BLO_TESTDATA_PATH, nav_shape=(2, 2, 2))
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


def test_incorrect_sig_shape(lt_ctx):
    sig_shape = (5, 5)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "blo",
            path=BLO_TESTDATA_PATH,
            sig_shape=sig_shape
        )
    assert e.match(
        r"sig_shape must be of size: 20736"
    )


def test_num_partitions(lt_ctx):
    ds = lt_ctx.load(
        "blo",
        path=BLO_TESTDATA_PATH,
        num_partitions=129,
    )
    assert len(list(ds.get_partitions())) == 129


def test_compare_backends(lt_ctx, default_blo, buffered_blo):
    y = random.choice(range(default_blo.shape.nav[0]))
    x = random.choice(range(default_blo.shape.nav[1]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_blo,
        x=x, y=y,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=buffered_blo,
        x=x, y=y,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="No support for direct I/O on Mac OS X"
)
def test_compare_direct_to_mmap(lt_ctx, default_blo, direct_blo):
    y = random.choice(range(default_blo.shape.nav[0]))
    x = random.choice(range(default_blo.shape.nav[1]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_blo,
        x=x, y=y,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=direct_blo,
        x=x, y=y,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_compare_backends_sparse(lt_ctx, default_blo, buffered_blo, as_sparse):
    roi = np.zeros(default_blo.shape.nav, dtype=bool).reshape((-1,))
    roi[0] = True
    roi[1] = True
    roi[16] = True
    roi[32] = True
    roi[-1] = True
    if as_sparse:
        roi = roi_as_sparse(roi)
    mm_f0 = lt_ctx.run_udf(dataset=default_blo, udf=PickUDF(), roi=roi)['intensity']
    buffered_f0 = lt_ctx.run_udf(dataset=buffered_blo, udf=PickUDF(), roi=roi)['intensity']

    assert np.allclose(mm_f0, buffered_f0)


def test_bad_params(ds_params_tester, standard_bad_ds_params):
    args = ("blo", BLO_TESTDATA_PATH)
    for params in standard_bad_ds_params:
        ds_params_tester(*args, **params)


@pytest.mark.skipif(not HAVE_BLO_16BIT_TESTDATA, reason="missing testdata")
def test_simple_open_16bit(blo_16bit: BloDataSet):
    assert tuple(blo_16bit.meta.shape) == (11, 7, 576, 576)
    assert blo_16bit.meta.raw_dtype == np.uint16


def test_pick_16bit(lt_ctx, blo_16bit):
    roi = np.zeros(blo_16bit.meta.shape.nav, dtype=bool).reshape((-1,))
    roi[0] = True
    frame = lt_ctx.run_udf(dataset=blo_16bit, udf=PickUDF(), roi=roi)['intensity'].data
    assert frame.sum() == 2055304
