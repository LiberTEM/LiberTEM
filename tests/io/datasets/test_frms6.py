import os
import pickle
import json
import random
import glob

import pytest
import numpy as np
import stemtool

from libertem.io.dataset.frms6 import (
    FRMS6DataSet, _map_y, FRMS6Decoder,
)
from libertem.analysis.raw import PickFrameAnalysis
from libertem.analysis.sum import SumAnalysis
from libertem.udf.sumsigudf import SumSigUDF
from libertem.io.dataset.base import TilingScheme, BufferedBackend, MMapBackend
from libertem.common import Shape
from libertem.common.buffers import reshaped_view
from libertem.udf.raw import PickUDF

from utils import (dataset_correction_verification, get_testdata_path,
    FakeBackend)

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


@pytest.fixture(scope='module')
def default_frms6_raw(tmpdir_factory):
    fn = tmpdir_factory.mktemp("data").join("frms6.raw")
    # we use a memory mapped file to make this work
    # on machines that can't hold the full dataset in memory
    data = np.memmap(str(fn), mode='w+', shape=(256, 256, 264, 264), dtype='uint16')
    view = reshaped_view(data, (256*256, 264, 264))
    root, ext = os.path.splitext(FRMS6_TESTDATA_PATH)
    files = list(sorted(glob.glob(root + '*.frms6')))
    blocksize = 15
    offset = 0
    # we skip the first file, it contains a zero reference
    for f in files[1:]:
        raw_shape = stemtool.util.pnccd.Frms6Reader.getDataShape(f)
        frame_count = raw_shape[-1]
        # We go blockwise to reduce memory consumption
        for start in range(0, frame_count, blocksize):
            stop = min(start+blocksize, frame_count)
            block = stemtool.util.pnccd.Frms6Reader.readData(
                f,
                image_range=(start, stop),
                pixels_x=raw_shape[0],
                pixels_y=raw_shape[1]
            )

            view[offset + start:offset + stop] = np.moveaxis(  # undo the transpose
                np.repeat(  # unbinning 4x in x direction
                    # invert lower half and attach right of upper half
                    # The detector consists of two chips that are arranged head-to-head
                    # The outputs of the two chips are just concatenated in the file, while LiberTEM
                    # re-assembles the data taking the spatial relation into account
                    np.concatenate((block[:264], np.flip(block[264:], axis=(0, 1,))), axis=1),
                    4, axis=1  # repeat options
                ),
                (0, 1, 2), (2, 1, 0)  # moveaxis options
            )
        offset += frame_count
    return data


def test_simple_open(default_frms6):
    assert tuple(default_frms6.shape) == (256, 256, 264, 264)


def test_auto_open_corrections_kwargs(lt_ctx):
    ds_corr = lt_ctx.load(
        'auto', path=FRMS6_TESTDATA_PATH, enable_offset_correction=True, nav_shape=(2, 3)
    )
    assert not np.allclose(ds_corr.get_correction_data().get_dark_frame(), 0)
    assert tuple(ds_corr.shape.nav) == (2, 3)
    assert isinstance(ds_corr, FRMS6DataSet)

    ds = lt_ctx.load(
        'auto', path=FRMS6_TESTDATA_PATH, enable_offset_correction=False, nav_shape=(2, 3)
    )
    assert not ds.get_correction_data().have_corrections()
    assert tuple(ds.shape.nav) == (2, 3)
    assert isinstance(ds, FRMS6DataSet)


def test_auto_open_corrections_posargs(lt_ctx):
    ds_corr = lt_ctx.load('auto', FRMS6_TESTDATA_PATH, True, None, None, (2, 3))
    assert not np.allclose(ds_corr.get_correction_data().get_dark_frame(), 0)
    assert tuple(ds_corr.shape.nav) == (2, 3)
    assert isinstance(ds_corr, FRMS6DataSet)

    ds = lt_ctx.load('auto', FRMS6_TESTDATA_PATH, False, None, None, (2, 3))
    assert not ds.get_correction_data().have_corrections()
    assert tuple(ds.shape.nav) == (2, 3)
    assert isinstance(ds, FRMS6DataSet)

    ds_corr = lt_ctx.load('frms6', FRMS6_TESTDATA_PATH, True, None, None, (2, 3))
    assert not np.allclose(ds_corr.get_correction_data().get_dark_frame(), 0)
    assert tuple(ds_corr.shape.nav) == (2, 3)
    assert isinstance(ds_corr, FRMS6DataSet)

    ds = lt_ctx.load('frms6', FRMS6_TESTDATA_PATH, False, None, None, (2, 3))
    assert not ds.get_correction_data().have_corrections()
    assert tuple(ds.shape.nav) == (2, 3)
    assert isinstance(ds, FRMS6DataSet)


def test_auto_uses_correct_backend(lt_ctx):
    with pytest.raises(RuntimeError):
        ds = lt_ctx.load(
            "auto",
            path=FRMS6_TESTDATA_PATH,
            io_backend=FakeBackend(),
        )
        lt_ctx.run_udf(
            dataset=ds,
            udf=SumSigUDF(),
        )


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


def test_positive_sync_offset(default_frms6_raw, lt_ctx):
    udf = SumSigUDF()
    sync_offset = 2

    ds = lt_ctx.load(
        "frms6", path=FRMS6_TESTDATA_PATH, nav_shape=(4, 2), sync_offset=sync_offset,
        enable_offset_correction=False
    )

    result = lt_ctx.run_udf(dataset=ds, udf=udf)
    result = result['intensity'].raw_data[:ds._meta.shape.nav.size - sync_offset]

    full_shape = default_frms6_raw.shape
    flat_nav = reshaped_view(default_frms6_raw, (-1, ) + full_shape[2:])
    cutout = flat_nav[2:8]
    ref = np.sum(cutout, axis=(-1, -2))

    print(ref.shape, result.shape)

    assert np.allclose(ref, result)


def test_negative_sync_offset(default_frms6_raw, lt_ctx):
    udf = SumSigUDF()
    sync_offset = -2

    ds = lt_ctx.load(
        "frms6", path=FRMS6_TESTDATA_PATH, nav_shape=(4, 2), sync_offset=sync_offset,
        enable_offset_correction=False
    )

    result = lt_ctx.run_udf(dataset=ds, udf=udf)
    result = result['intensity'].raw_data[abs(sync_offset):]

    full_shape = default_frms6_raw.shape
    flat_nav = reshaped_view(default_frms6_raw, (-1, ) + full_shape[2:])
    cutout = flat_nav[:6]
    ref = np.sum(cutout, axis=(-1, -2))

    print(ref.shape, result.shape)

    assert np.allclose(ref, result)


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


def test_reshape_nav(default_frms6_raw, lt_ctx):
    udf = SumSigUDF()

    full_shape = default_frms6_raw.shape
    flat_nav = reshaped_view(default_frms6_raw, (-1, ) + full_shape[2:])

    ds_1 = lt_ctx.load(
        "frms6",
        path=FRMS6_TESTDATA_PATH,
        nav_shape=(8,),
        enable_offset_correction=False
    )
    result_1 = lt_ctx.run_udf(dataset=ds_1, udf=udf)
    ref_1 = flat_nav[:8].sum(axis=(-1, -2))
    assert np.allclose(result_1['intensity'].data, ref_1)

    ds_2 = lt_ctx.load(
        "frms6",
        path=FRMS6_TESTDATA_PATH,
        nav_shape=(2, 2, 2),
        enable_offset_correction=False
    )
    result_2 = lt_ctx.run_udf(dataset=ds_2, udf=udf)
    ref_2 = flat_nav[:8].sum(axis=(-1, -2)).reshape((2, 2, 2))
    assert np.allclose(result_2['intensity'].data, ref_2)


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
