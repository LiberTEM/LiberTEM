import os
import pickle
import json
import hashlib
import random

import pytest
import numpy as np

from libertem.io.dataset.frms6 import (
    FRMS6DataSet, _map_y, FRMS6Decoder,
)
from libertem.analysis.raw import PickFrameAnalysis
from libertem.analysis.sum import SumAnalysis
from libertem.udf.sumsigudf import SumSigUDF
from libertem.io.dataset.base import TilingScheme, BufferedBackend, MMapBackend
from libertem.common import Shape
from libertem.udf.raw import PickUDF

from utils import dataset_correction_verification, get_testdata_path

FRMS6_TESTDATA_PATH = os.path.join(get_testdata_path(), 'frms6', 'C16_15_24_151203_019.hdr')
HAVE_FRMS6_TESTDATA = os.path.exists(FRMS6_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_FRMS6_TESTDATA, reason="need frms6 testdata")  # NOQA


@pytest.fixture
def default_frms6(lt_ctx):
    ds = lt_ctx.load(
        "frms6",
        path=FRMS6_TESTDATA_PATH,
        io_backend=MMapBackend(),
    )
    return ds


@pytest.fixture
def buffered_frms6(lt_ctx):
    buffered = BufferedBackend()
    return lt_ctx.load(
        "frms6",
        path=str(FRMS6_TESTDATA_PATH),
        io_backend=buffered,
    )


def test_simple_open(default_frms6):
    assert tuple(default_frms6.shape) == (256, 256, 264, 264)


def test_detetct(lt_ctx):
    assert FRMS6DataSet.detect_params(
        FRMS6_TESTDATA_PATH, lt_ctx.executor
    )["parameters"] is not False


def test_check_valid(default_frms6):
    default_frms6.check_valid()


def test_sum_analysis(default_frms6, lt_ctx):
    roi = {
        "shape": "disk",
        "cx": 5,
        "cy": 6,
        "r": 7,
    }
    analysis = SumAnalysis(dataset=default_frms6, parameters={
        "roi": roi,
    })
    # not checking result yet, just making sure it doesn't crash:
    lt_ctx.run(analysis)


def test_pick_job(default_frms6, lt_ctx):
    analysis = lt_ctx.create_pick_job(dataset=default_frms6, origin=(16, 16))
    results = lt_ctx.run(analysis)
    assert results.shape == (264, 264)


@pytest.mark.parametrize(
    'TYPE', ['JOB', 'UDF']
)
def test_pick_analysis(default_frms6, lt_ctx, TYPE):
    analysis = PickFrameAnalysis(dataset=default_frms6, parameters={"x": 16, "y": 16})
    analysis.TYPE = TYPE
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (264, 264)


@pytest.mark.parametrize(
    # Default is too large for test without ROI
    "with_roi", (True, )
)
def test_correction(default_frms6, lt_ctx, with_roi):
    ds = default_frms6

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


def test_pickle_is_small(default_frms6):
    pickled = pickle.dumps(default_frms6)
    pickle.loads(pickled)

    # because of the dark frame stuff, the dataset is actually quite large:
    assert len(pickled) < 300 * 1024


def test_cache_key_json_serializable(default_frms6):
    json.dumps(default_frms6.get_cache_key())


@pytest.mark.dist
def test_dist_process(default_frms6, dist_ctx):
    roi = {
        "shape": "disk",
        "cx": 5,
        "cy": 6,
        "r": 7,
    }
    analysis = SumAnalysis(dataset=default_frms6, parameters={"roi": roi})
    dist_ctx.run(analysis)


@pytest.mark.dist
def test_initialize(default_frms6, dist_ctx):
    assert default_frms6._filenames is not None
    assert default_frms6._hdr_info is not None
    assert default_frms6._hdr_info is not None

# TODO: gain map tests
# TODO: test load request message
# TODO: test error conditions


def test_map_y():
    assert _map_y(y=0, xs=264, binning=4, num_rows=264) == (0, 0)
    assert _map_y(y=32, xs=264, binning=4, num_rows=264) == (32, 0)
    assert _map_y(y=33, xs=264, binning=4, num_rows=264) == (32, 264)
    assert _map_y(y=65, xs=264, binning=4, num_rows=264) == (0, 264)

    assert _map_y(y=0, xs=264, binning=1, num_rows=264) == (0, 0)
    assert _map_y(y=32, xs=264, binning=1, num_rows=264) == (32, 0)
    assert _map_y(y=33, xs=264, binning=1, num_rows=264) == (33, 0)
    assert _map_y(y=65, xs=264, binning=1, num_rows=264) == (65, 0)
    assert _map_y(y=131, xs=264, binning=1, num_rows=264) == (131, 0)
    assert _map_y(y=132, xs=264, binning=1, num_rows=264) == (131, 264)
    assert _map_y(y=263, xs=264, binning=1, num_rows=264) == (0, 264)


@pytest.mark.with_numba
@pytest.mark.parametrize(
    'binning', [1, 2, 4],
)
def test_decode(binning):
    out = np.zeros((8, 8, 264), dtype=np.uint16)
    reads = [
        np.random.randint(low=1, high=1024, size=(1, 264), dtype=np.uint16)
        for i in range(out.shape[0] * out.shape[1] // binning)
    ]

    decoder = FRMS6Decoder(binning=binning)
    decode = decoder.get_decode(native_dtype="u2", read_dtype=np.float32)

    for idx, read in enumerate(reads):
        decode(
            inp=read,
            out=out,
            idx=idx,
            native_dtype=np.uint16,
            rr=None,
            origin=np.array([0, 0, 0]),
            shape=np.array(out.shape),
            ds_shape=np.array([1024, 264, 264]),
        )

    for idx, px in enumerate(out.reshape((-1,))):
        assert not np.isclose(px, 0)


@pytest.mark.with_numba
def test_with_roi(default_frms6, lt_ctx):
    udf = PickUDF()
    roi = np.zeros(default_frms6.shape.nav, dtype=bool)
    roi[0] = 1
    res = lt_ctx.run_udf(udf=udf, dataset=default_frms6, roi=roi)
    np.array(res['intensity']).shape == (1, 256, 256)


def test_read_invalid_tileshape(default_frms6):
    partitions = default_frms6.get_partitions()
    p = next(partitions)

    tileshape = Shape(
        (1, 3, 264),
        sig_dims=2,
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_frms6.shape,
    )

    with pytest.raises(ValueError):
        next(p.get_tiles(tiling_scheme=tiling_scheme))


def test_positive_sync_offset(lt_ctx):
    udf = SumSigUDF()
    sync_offset = 2

    ds = lt_ctx.load(
        "frms6", path=FRMS6_TESTDATA_PATH, nav_shape=(4, 2), sync_offset=sync_offset
    )

    result = lt_ctx.run_udf(dataset=ds, udf=udf)
    result = result['intensity'].raw_data[:ds._meta.shape.nav.size - sync_offset]
    result_sha1 = hashlib.sha1()
    result_sha1.update(result)

    # how to generate the hash below
    # ds = ctx.load("frms6", path=FRMS6_TESTDATA_PATH, nav_shape=(4, 2))
    # udf = SumSigUDF()
    # result = ctx.run_udf(dataset=ds, udf=udf)
    # sha1 = hashlib.sha1()
    # sha1.update(result['intensity'].raw_data[sync_offset:])
    # sha1.hexdigest()
    assert result_sha1.hexdigest() == "460278543d8dbbed5080c56450c8669136750b78"


def test_negative_sync_offset(default_frms6, lt_ctx):
    udf = SumSigUDF()
    sync_offset = -2

    ds = lt_ctx.load(
        "frms6", path=FRMS6_TESTDATA_PATH, nav_shape=(4, 2), sync_offset=sync_offset
    )

    result = lt_ctx.run_udf(dataset=ds, udf=udf)
    result = result['intensity'].raw_data[abs(sync_offset):]
    result_sha1 = hashlib.sha1()
    result_sha1.update(result)

    # how to generate the hash below
    # ds = ctx.load("frms6", path=FRMS6_TESTDATA_PATH, nav_shape=(4, 2))
    # udf = SumSigUDF()
    # result = ctx.run_udf(dataset=ds, udf=udf)
    # sha1 = hashlib.sha1()
    # sha1.update(result['intensity'].raw_data[:ds._meta.shape.nav.size + sync_offset])
    # sha1.hexdigest()
    assert result_sha1.hexdigest() == "3bc6f0abf253f08bd05f6de5fec6403f09d94b49"


def test_offset_smaller_than_image_count(lt_ctx):
    sync_offset = -65540

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "frms6",
            path=FRMS6_TESTDATA_PATH,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-65536, 65536\), which is \(-image_count, image_count\)"
    )


def test_offset_greater_than_image_count(lt_ctx):
    sync_offset = 65540

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "frms6",
            path=FRMS6_TESTDATA_PATH,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-65536, 65536\), which is \(-image_count, image_count\)"
    )


def test_reshape_nav(lt_ctx):
    udf = SumSigUDF()

    # how to generate the hash below
    # ds = ctx.load("frms6", path=FRMS6_TESTDATA_PATH, nav_shape=(4, 2))
    # udf = SumSigUDF()
    # result = ctx.run_udf(dataset=ds, udf=udf)
    # sha1 = hashlib.sha1()
    # sha1.update(result['intensity'].raw_data)
    # sha1.hexdigest()

    ds_1 = lt_ctx.load("frms6", path=FRMS6_TESTDATA_PATH, nav_shape=(8,))
    result_1 = lt_ctx.run_udf(dataset=ds_1, udf=udf)

    result_1_sha1 = hashlib.sha1()
    result_1_sha1.update(result_1['intensity'].raw_data)
    assert result_1_sha1.hexdigest() == "ae917373ac2fc15e13903f8d2a07b0545dd59a87"

    ds_2 = lt_ctx.load("frms6", path=FRMS6_TESTDATA_PATH, nav_shape=(2, 2, 2))
    result_2 = lt_ctx.run_udf(dataset=ds_2, udf=udf)

    result_2_sha1 = hashlib.sha1()
    result_2_sha1.update(result_2['intensity'].raw_data)
    assert result_2_sha1.hexdigest() == "ae917373ac2fc15e13903f8d2a07b0545dd59a87"


def test_incorrect_sig_shape(lt_ctx):
    sig_shape = (5, 5)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "frms6",
            path=FRMS6_TESTDATA_PATH,
            sig_shape=sig_shape
        )
    assert e.match(
        r"sig_shape must be of size: 69696"
    )


def test_compare_backends(lt_ctx, default_frms6, buffered_frms6):
    y = random.choice(range(default_frms6.shape.nav[0]))
    x = random.choice(range(default_frms6.shape.nav[1]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_frms6,
        x=x, y=y,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=buffered_frms6,
        x=x, y=y,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


def test_compare_backends_sparse(lt_ctx, default_frms6, buffered_frms6):
    roi = np.zeros(default_frms6.shape.nav, dtype=np.bool).reshape((-1,))
    roi[0] = True
    roi[1] = True
    roi[16] = True
    roi[32] = True
    roi[-1] = True
    mm_f0 = lt_ctx.run_udf(dataset=default_frms6, udf=PickUDF(), roi=roi)['intensity']
    buffered_f0 = lt_ctx.run_udf(dataset=buffered_frms6, udf=PickUDF(), roi=roi)['intensity']

    assert np.allclose(mm_f0, buffered_f0)
