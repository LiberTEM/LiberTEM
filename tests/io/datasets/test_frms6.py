import os
import pickle
import json

import pytest
import numpy as np

from libertem.io.dataset.frms6 import (
    FRMS6DataSet, _map_y, FRMS6Decoder,
)
from libertem.analysis.raw import PickFrameAnalysis
from libertem.analysis.sum import SumAnalysis
from libertem.io.dataset.base import TilingScheme
from libertem.common import Shape
from libertem.udf.raw import PickUDF

FRMS6_TESTDATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                                 'data', 'frms6', 'C16_15_24_151203_019.hdr')
HAVE_FRMS6_TESTDATA = os.path.exists(FRMS6_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_FRMS6_TESTDATA, reason="need frms6 testdata")  # NOQA


@pytest.fixture
def default_frms6(lt_ctx):
    ds = FRMS6DataSet(path=FRMS6_TESTDATA_PATH)
    ds = ds.initialize(lt_ctx.executor)
    return ds


@pytest.fixture
def dist_frms6(dist_ctx):
    path = "/data/frms6/C16_15_24_151203_019.hdr"
    ds = FRMS6DataSet(path=path)
    ds = ds.initialize(dist_ctx.executor)
    return ds


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


def test_pickle_is_small(default_frms6):
    pickled = pickle.dumps(default_frms6)
    pickle.loads(pickled)

    # because of the dark frame stuff, the dataset is actually quite large:
    assert len(pickled) < 300 * 1024


def test_cache_key_json_serializable(default_frms6):
    json.dumps(default_frms6.get_cache_key())


@pytest.mark.dist
def test_dist_process(dist_frms6, dist_ctx):
    roi = {
        "shape": "disk",
        "cx": 5,
        "cy": 6,
        "r": 7,
    }
    analysis = SumAnalysis(dataset=dist_frms6, parameters={"roi": roi})
    dist_ctx.run(analysis)


@pytest.mark.dist
def test_initialize(dist_frms6, dist_ctx):
    assert dist_frms6._filenames is not None
    assert dist_frms6._hdr_info is not None
    assert dist_frms6._hdr_info is not None

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
