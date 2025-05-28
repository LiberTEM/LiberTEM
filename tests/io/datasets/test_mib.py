import os
import pathlib
import sys
import json
from unittest import mock
import random
import warnings

import numpy as np
import pytest
import cloudpickle

from libertem.io.dataset.mib import MIBDataSet, get_filenames
from libertem.udf.raw import PickUDF
from libertem.analysis.raw import PickFrameAnalysis
from libertem.common import Shape
from libertem.common.buffers import reshaped_view
from libertem.udf.sumsigudf import SumSigUDF
from libertem.io.dataset.base import (
    TilingScheme, BufferedBackend, MMapBackend, DirectBackend
)

from utils import dataset_correction_verification, get_testdata_path, ValidationUDF, roi_as_sparse

try:
    import rsciio
except ModuleNotFoundError:
    rsciio = None


MIB_TESTDATA_PATH = os.path.join(get_testdata_path(), 'default.mib')
HAVE_MIB_TESTDATA = os.path.exists(MIB_TESTDATA_PATH)

needsdata = pytest.mark.skipif(not HAVE_MIB_TESTDATA, reason="need .mib testdata")  # NOQA


@pytest.fixture
def default_mib(lt_ctx):
    nav_shape = (32, 32)
    ds = lt_ctx.load(
        "mib",
        path=MIB_TESTDATA_PATH,
        nav_shape=nav_shape,
        io_backend=MMapBackend(),
        num_partitions=4,
    )
    return ds


@pytest.fixture
def default_mib_readahead(lt_ctx):
    nav_shape = (32, 32)
    ds = lt_ctx.load(
        "mib",
        path=MIB_TESTDATA_PATH,
        nav_shape=nav_shape,
        io_backend=MMapBackend(enable_readahead_hints=True),
        num_partitions=4,
    )
    return ds


@pytest.fixture
def buffered_mib(lt_ctx):
    buffered = BufferedBackend()
    ds = lt_ctx.load(
        "mib",
        path=MIB_TESTDATA_PATH,
        nav_shape=(32, 32),
        io_backend=buffered,
    )
    return ds


@pytest.fixture
def direct_mib(lt_ctx):
    direct = DirectBackend()
    ds = lt_ctx.load(
        "mib",
        path=MIB_TESTDATA_PATH,
        nav_shape=(32, 32),
        io_backend=direct,
    )
    return ds


@pytest.fixture(scope='module')
def default_mib_raw():
    from rsciio.quantumdetector import load_mib_data
    data = load_mib_data(MIB_TESTDATA_PATH, lazy=True)
    shape = (32, 32, 256, 256)
    return data.reshape(shape)


def test_glob(tmp_path, naughty_filename):
    naughty_dir = tmp_path / naughty_filename
    naughty_dir.mkdir()

    naughty_hdr = naughty_dir / (naughty_filename + '.hdr')
    naughty_hdr.touch()

    naughty_mib_0 = naughty_dir / (naughty_filename + '1.mib')
    naughty_mib_1 = naughty_dir / (naughty_filename + '2.mib')
    naughty_mib_0.touch()
    naughty_mib_1.touch()

    filenames_1 = set(get_filenames(naughty_hdr))
    filenames_2 = set(get_filenames(naughty_mib_0))
    print(filenames_1)
    print(filenames_2)
    target = {str(naughty_mib_0), str(naughty_mib_1)}
    assert filenames_1 == target
    assert filenames_2 == target


@needsdata
def test_detect(lt_ctx):
    params = MIBDataSet.detect_params(MIB_TESTDATA_PATH, lt_ctx.executor)["parameters"]
    assert params == {
        "path": MIB_TESTDATA_PATH,
        "nav_shape": (32, 32),
        "sig_shape": (256, 256)
    }


@needsdata
def test_simple_open(default_mib):
    assert tuple(default_mib.shape) == (32, 32, 256, 256)


@needsdata
def test_positive_sync_offset(default_mib, lt_ctx):
    udf = SumSigUDF()
    sync_offset = 2

    ds_with_offset = MIBDataSet(
        path=MIB_TESTDATA_PATH, nav_shape=(32, 32), sync_offset=sync_offset,
        num_partitions=4,
    )
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == 2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (16,) + tuple(ds_with_offset.shape.sig),
        sig_dims=ds_with_offset.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds_with_offset.shape,
    )

    tiles = p0.get_tiles(tiling_scheme)
    t0 = next(tiles)
    assert tuple(t0.tile_slice.origin) == (0, 0, 0)
    for t in tiles:
        pass

    for p in ds_with_offset.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p.slice.origin == (768, 0, 0)
    assert p.slice.shape[0] == 256

    result = lt_ctx.run_udf(dataset=default_mib, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
        :ds_with_offset._meta.image_count - sync_offset
    ]

    assert np.allclose(result, result_with_offset)


@needsdata
def test_negative_sync_offset(default_mib, lt_ctx):
    udf = SumSigUDF()
    sync_offset = -2

    ds_with_offset = MIBDataSet(
        path=MIB_TESTDATA_PATH, nav_shape=(32, 32), sync_offset=sync_offset,
        num_partitions=4,
    )
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == -2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (16,) + tuple(ds_with_offset.shape.sig),
        sig_dims=ds_with_offset.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds_with_offset.shape,
    )

    tiles = p0.get_tiles(tiling_scheme)
    t0 = next(tiles)
    assert tuple(t0.tile_slice.origin) == (2, 0, 0)
    for t in tiles:
        pass

    for p in ds_with_offset.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p.slice.origin == (768, 0, 0)
    assert p.slice.shape[0] == 256

    result = lt_ctx.run_udf(dataset=default_mib, udf=udf)
    result = result['intensity'].raw_data[:default_mib._meta.image_count - abs(sync_offset)]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


@needsdata
def test_offset_smaller_than_nav_shape(lt_ctx):
    nav_shape = (32, 32)
    sync_offset = -1030

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "mib",
            path=MIB_TESTDATA_PATH,
            nav_shape=nav_shape,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-1024, 1024\), which is \(-image_count, image_count\)"
        )


@needsdata
def test_offset_greater_than_nav_shape(lt_ctx):
    nav_shape = (32, 32)
    sync_offset = 1030

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "mib",
            path=MIB_TESTDATA_PATH,
            nav_shape=nav_shape,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-1024, 1024\), which is \(-image_count, image_count\)"
        )


@needsdata
@pytest.mark.with_numba
def test_read(default_mib):
    partitions = default_mib.get_partitions()
    p = next(partitions)
    assert len(p.shape) == 3
    assert tuple(p.shape[1:]) == (256, 256)

    tileshape = Shape(
        (3,) + tuple(default_mib.shape.sig),
        sig_dims=default_mib.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_mib.shape,
    )

    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    # we get 3D tiles here, because MIB partitions are inherently 3D
    assert tuple(t.tile_slice.shape) == (3, 256, 256)
    for _ in tiles:
        pass


@needsdata
def test_scheme_too_large(default_mib):
    partitions = default_mib.get_partitions()
    p = next(partitions)
    depth = p.shape[0]

    # we make a tileshape that is too large for the partition here:
    tileshape = Shape(
        (depth + 1,) + tuple(default_mib.shape.sig),
        sig_dims=default_mib.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_mib.shape,
    )

    # tile shape is clamped to partition shape:
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == tuple((depth,) + default_mib.shape.sig)
    for _ in tiles:
        pass


@needsdata
@pytest.mark.with_numba
def test_read_ahead(default_mib_readahead):
    partitions = default_mib_readahead.get_partitions()
    p = next(partitions)
    assert len(p.shape) == 3
    assert tuple(p.shape[1:]) == (256, 256)

    tileshape = Shape(
        (3,) + tuple(default_mib_readahead.shape.sig),
        sig_dims=default_mib_readahead.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_mib_readahead.shape,
    )

    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    # we get 3D tiles here, because MIB partitions are inherently 3D
    assert tuple(t.tile_slice.shape) == (3, 256, 256)
    for _ in tiles:
        pass


@needsdata
@pytest.mark.skipif(rsciio is None, reason="No rosettasciio found")
def test_comparison(default_mib, default_mib_raw, lt_ctx_fast):
    udf = ValidationUDF(
        reference=reshaped_view(default_mib_raw, (-1, *tuple(default_mib.shape.sig)))
    )
    lt_ctx_fast.run_udf(udf=udf, dataset=default_mib)


@needsdata
@pytest.mark.skipif(rsciio is None, reason="No rosettasciio found")
def test_comparison_roi(default_mib, default_mib_raw, lt_ctx_fast):
    roi = np.random.choice(
        [True, False],
        size=tuple(default_mib.shape.nav),
        p=[0.1, 0.9]
    )
    udf = ValidationUDF(reference=default_mib_raw[roi])
    lt_ctx_fast.run_udf(udf=udf, dataset=default_mib, roi=roi)


@needsdata
def test_pickle_is_small(default_mib):
    pickled = cloudpickle.dumps(default_mib)
    cloudpickle.loads(pickled)

    # let's keep the pickled dataset size small-ish:
    assert len(pickled) < 2 * 1024


@needsdata
def test_apply_mask_analysis(default_mib, lt_ctx):
    mask = np.ones((256, 256))
    analysis = lt_ctx.create_mask_analysis(factories=[lambda: mask], dataset=default_mib)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (32, 32)


@needsdata
def test_sum_analysis(default_mib, lt_ctx):
    analysis = lt_ctx.create_sum_analysis(dataset=default_mib)
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (256, 256)


@needsdata
def test_pick_analysis(default_mib, lt_ctx):
    analysis = PickFrameAnalysis(dataset=default_mib, parameters={"x": 16, "y": 16})
    results = lt_ctx.run(analysis)
    assert results[0].raw_data.shape == (256, 256)


@needsdata
@pytest.mark.parametrize(
    "with_roi", (True, False)
)
@pytest.mark.slow
def test_correction(default_mib, lt_ctx, with_roi):
    ds = default_mib

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None
    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx, exclude=[(45, 144), (124, 30)])
    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)


@needsdata
@pytest.mark.with_numba
def test_with_roi(default_mib, lt_ctx):
    udf = PickUDF()
    roi = np.zeros(default_mib.shape.nav, dtype=bool)
    roi[0] = 1
    res = lt_ctx.run_udf(udf=udf, dataset=default_mib, roi=roi)
    np.array(res['intensity']).shape == (1, 256, 256)


@needsdata
def test_read_at_boundaries(default_mib, lt_ctx):
    nav_shape = (32, 32)
    ds_odd = MIBDataSet(path=MIB_TESTDATA_PATH, nav_shape=nav_shape)
    ds_odd = ds_odd.initialize(lt_ctx.executor)

    sumjob_odd = lt_ctx.create_sum_analysis(dataset=ds_odd)
    res_odd = lt_ctx.run(sumjob_odd)

    sumjob = lt_ctx.create_sum_analysis(dataset=default_mib)
    res = lt_ctx.run(sumjob)

    assert np.allclose(res[0].raw_data, res_odd[0].raw_data)


@needsdata
def test_diagnostics(default_mib):
    print(default_mib.diagnostics)


@needsdata
def test_cache_key_json_serializable(default_mib):
    json.dumps(default_mib.get_cache_key())


@needsdata
@pytest.mark.dist
def test_mib_dist(dist_ctx):
    nav_shape = (32, 32)
    ds = MIBDataSet(path=MIB_TESTDATA_PATH, nav_shape=nav_shape)
    ds = ds.initialize(dist_ctx.executor)
    analysis = dist_ctx.create_sum_analysis(dataset=ds)
    results = dist_ctx.run(analysis)
    assert results[0].raw_data.shape == (256, 256)


@needsdata
def test_too_many_files(lt_ctx):
    ds = MIBDataSet(path=MIB_TESTDATA_PATH, nav_shape=(32, 32))

    with mock.patch('libertem.io.dataset.mib.glob', side_effect=lambda p: [
            "/a/%d.mib" % i
            for i in range(256*256)
    ]):
        with pytest.warns(RuntimeWarning) as record:
            ds._filenames()

    assert len(record) == 1
    assert "Saving data in many small files" in record[0].message.args[0]


@needsdata
def test_not_too_many_files(lt_ctx):
    ds = MIBDataSet(path=MIB_TESTDATA_PATH, nav_shape=(32, 32))

    with mock.patch('libertem.io.dataset.mib.glob', side_effect=lambda p: [
            "/a/%d.mib" % i
            for i in range(256)
    ]):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            ds._filenames()


@needsdata
def test_reshape_nav(lt_ctx):
    udf = SumSigUDF()

    ds_with_1d_nav = lt_ctx.load("mib", path=MIB_TESTDATA_PATH, nav_shape=(8,))
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    ds_with_2d_nav = lt_ctx.load("mib", path=MIB_TESTDATA_PATH, nav_shape=(4, 2))
    result_with_2d_nav = lt_ctx.run_udf(dataset=ds_with_2d_nav, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load("mib", path=MIB_TESTDATA_PATH, nav_shape=(2, 2, 2))
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


@needsdata
def test_incorrect_sig_shape(lt_ctx):
    nav_shape = (32, 32)
    sig_shape = (5, 5)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "mib",
            path=MIB_TESTDATA_PATH,
            nav_shape=nav_shape,
            sig_shape=sig_shape
        )
    assert e.match(
        r"sig_shape must be of size: 65536"
    )


@needsdata
def test_scan_size_deprecation(lt_ctx):
    scan_size = (2, 2)

    with pytest.warns(FutureWarning):
        ds = lt_ctx.load(
            "mib",
            path=MIB_TESTDATA_PATH,
            scan_size=scan_size,
        )
    assert tuple(ds.shape) == (2, 2, 256, 256)


@needsdata
def test_compare_backends(lt_ctx, default_mib, buffered_mib):
    y = random.choice(range(default_mib.shape.nav[0]))
    x = random.choice(range(default_mib.shape.nav[1]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_mib,
        x=x, y=y,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=buffered_mib,
        x=x, y=y,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


@needsdata
@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="No support for direct I/O on Mac OS X"
)
def test_compare_direct_to_mmap(lt_ctx, default_mib, direct_mib):
    y = random.choice(range(default_mib.shape.nav[0]))
    x = random.choice(range(default_mib.shape.nav[1]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_mib,
        x=x, y=y,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=direct_mib,
        x=x, y=y,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


@needsdata
@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_compare_backends_sparse(lt_ctx, default_mib, buffered_mib, as_sparse):
    roi = np.zeros(default_mib.shape.nav, dtype=bool).reshape((-1,))
    roi[0] = True
    roi[1] = True
    roi[16] = True
    roi[32] = True
    roi[-1] = True
    if as_sparse:
        roi = roi_as_sparse(roi)
    mm_f0 = lt_ctx.run_udf(dataset=default_mib, udf=PickUDF(), roi=roi)['intensity']
    buffered_f0 = lt_ctx.run_udf(dataset=buffered_mib, udf=PickUDF(), roi=roi)['intensity']

    assert np.allclose(mm_f0, buffered_f0)


@needsdata
@pytest.mark.slow
def test_run_many_files(lt_ctx):
    # import tarfile
    # archive = pathlib.Path(MIB_TESTDATA_PATH).parent / 'many_mib.tar.gz'
    # outdir = pathlib.Path(tmpdir_factory.mktemp('many_mib_files'))

    # with tarfile.open(archive, mode='r:gz') as tp:
    #     tp.extractall(path=outdir)

    out_dir = pathlib.Path(MIB_TESTDATA_PATH).parent / 'many_mib'
    hdr_path = out_dir / 'w_140 h_139-2map.hdr'
    nav_shape = (139, 141)
    lt_ctx.load('mib', path=hdr_path, nav_shape=nav_shape)


def test_bad_params(ds_params_tester, standard_bad_ds_params):
    args = ("mib", MIB_TESTDATA_PATH)
    for params in standard_bad_ds_params:
        if 'nav_shape' not in params:
            params['nav_shape'] = (1, 1)
        ds_params_tester(*args, **params)


@needsdata
def test_read_scan_shape_header(lt_ctx):
    out_dir = pathlib.Path(MIB_TESTDATA_PATH).parent / 'merlin_scan_header/'
    hdr_path = (
        out_dir / 't2_200kV_alpha3_28_20_px_5_4nm_300k_C2_30um_CL_10m_1ms_1L_with_flatfield.hdr'
    )
    ds = lt_ctx.load('mib', path=hdr_path)
    assert ds.shape.nav.to_tuple() == (20, 28)
    assert ds.shape.sig.to_tuple() == (514, 514)


@needsdata
def test_no_num_partitions(lt_ctx):
    ds = lt_ctx.load(
        "mib",
        path=MIB_TESTDATA_PATH,
        nav_shape=(32, 32),
    )
    lt_ctx.run_udf(dataset=ds, udf=SumSigUDF())
