import os
import sys
from glob import glob
import random

import numpy as np
import pytest

from libertem.io.dataset.dm import DMDataSet
from libertem.udf.sum import SumUDF
from libertem.common import Shape
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.raw import PickUDF
from libertem.io.dataset.base import (
    TilingScheme, BufferedBackend, MMapBackend, DataSetException, DirectBackend
)

from utils import dataset_correction_verification, get_testdata_path, ValidationUDF, roi_as_sparse

try:
    import hyperspy.api as hs
except ModuleNotFoundError:
    hs = None

DM_TESTDATA_PATH = os.path.join(get_testdata_path(), 'dm')
HAVE_DM_TESTDATA = os.path.exists(DM_TESTDATA_PATH)

pytestmark = pytest.mark.skipif(not HAVE_DM_TESTDATA, reason="need .dm4 testdata")  # NOQA


@pytest.fixture(scope='module')
def dm_stack_glob():
    return glob(os.path.join(DM_TESTDATA_PATH, '2018-7-17*.dm4'))


@pytest.fixture(scope='module')
def dm_3d_glob():
    return glob(os.path.join(DM_TESTDATA_PATH, '3D', '*.dm3'))


@pytest.fixture
def default_dm(lt_ctx, dm_stack_glob):
    files = list(sorted(dm_stack_glob))
    ds = lt_ctx.load(
        "dm",
        files=files,
        io_backend=MMapBackend(),
    )
    return ds


@pytest.fixture(scope='module')
def default_dm_raw(dm_stack_glob):
    files = list(sorted(dm_stack_glob))
    return np.stack([hs.load(file).data for file in files])


@pytest.fixture
def buffered_dm(lt_ctx, dm_stack_glob):
    buffered = BufferedBackend()
    files = list(sorted(dm_stack_glob))
    return lt_ctx.load(
        "dm",
        files=files,
        io_backend=buffered,
    )


@pytest.fixture
def direct_dm(lt_ctx, dm_stack_glob):
    buffered = DirectBackend(dm_stack_glob)
    files = list(sorted(dm_stack_glob))
    return lt_ctx.load(
        "dm",
        files=files,
        io_backend=buffered,
    )


@pytest.fixture
def dm_stack_of_3d(lt_ctx, dm_3d_glob):
    files = list(sorted(dm_3d_glob))
    ds = lt_ctx.load("dm", files=files)
    return ds


@pytest.fixture(scope='module')
def default_dm_3d_raw(dm_3d_glob):
    files = list(sorted(dm_3d_glob))
    return np.concatenate([hs.load(file).data for file in files], axis=0)


def test_simple_open(default_dm):
    assert tuple(default_dm.shape) == (10, 3838, 3710)


def test_check_valid(default_dm):
    default_dm.check_valid()


@pytest.mark.skipif(hs is None, reason="No HyperSpy found")
def test_comparison(default_dm, default_dm_raw, lt_ctx_fast):
    udf = ValidationUDF(reference=default_dm_raw)
    lt_ctx_fast.run_udf(udf=udf, dataset=default_dm)


@pytest.mark.skipif(hs is None, reason="No HyperSpy found")
def test_comparison_roi(default_dm, default_dm_raw, lt_ctx_fast):
    roi = np.random.choice(
        [True, False],
        size=tuple(default_dm.shape.nav),
        p=[0.5, 0.5]
    )
    udf = ValidationUDF(reference=default_dm_raw[roi])
    lt_ctx_fast.run_udf(udf=udf, dataset=default_dm, roi=roi)


@pytest.mark.skipif(hs is None, reason="No HyperSpy found")
def test_comparison_3d(dm_stack_of_3d, default_dm_3d_raw, lt_ctx_fast):
    udf = ValidationUDF(reference=default_dm_3d_raw)
    lt_ctx_fast.run_udf(udf=udf, dataset=dm_stack_of_3d)


@pytest.mark.skipif(hs is None, reason="No HyperSpy found")
def test_comparison_3d_roi(dm_stack_of_3d, default_dm_3d_raw, lt_ctx_fast):
    roi = np.random.choice(
        [True, False],
        size=tuple(dm_stack_of_3d.shape.nav),
        p=[0.5, 0.5]
    )
    udf = ValidationUDF(reference=default_dm_3d_raw[roi])
    lt_ctx_fast.run_udf(udf=udf, dataset=dm_stack_of_3d, roi=roi)


@pytest.mark.parametrize(
    "with_roi", (True, False)
)
@pytest.mark.slow
def test_correction(default_dm, lt_ctx, with_roi):
    ds = default_dm

    if with_roi:
        roi = np.zeros(ds.shape.nav, dtype=bool)
        roi[:1] = True
    else:
        roi = None

    dataset_correction_verification(ds=ds, roi=roi, lt_ctx=lt_ctx)

# Disabled while DMDataSet.detect_params returns False
# to disable it in the Web UI
# def test_detect_1(lt_ctx):
#     fpath = os.path.join(DM_TESTDATA_PATH, '2018-7-17 15_29_0000.dm4')
#     assert DMDataSet.detect_params(
#         path=fpath,
#         executor=lt_ctx.executor,
#     )["parameters"] == {
#         'files': [fpath],
#     }


def test_get_metadata():
    import distributed
    from libertem.api import Context
    from libertem.io.dataset.dm import _get_metadata

    with distributed.Client(
            n_workers=2,
            threads_per_worker=1,
            processes=False
    ) as _:
        ctx = Context.make_with("dask-integration")
        fpath = os.path.join(DM_TESTDATA_PATH, '2018-7-17 15_29_0000.dm4')
        ctx.executor.run_function(lambda: _get_metadata(fpath))


def test_get_metadata_2():
    from libertem.io.dataset.dm import _get_metadata

    fpath = os.path.join(DM_TESTDATA_PATH, '2018-7-17 15_29_0000.dm4')
    _get_metadata(fpath)


def test_detect_2(lt_ctx):
    assert DMDataSet.detect_params(
        path="nofile.someext",
        executor=lt_ctx.executor,
    ) is False


def test_same_offset(lt_ctx, dm_stack_glob):
    files = dm_stack_glob
    ds = lt_ctx.load("dm", files=files, same_offset=True)
    ds.check_valid()


def test_repr(default_dm):
    assert repr(default_dm) == "<DMDataSet for a stack of 10 files>"


@pytest.mark.dist
def test_dm_dist(dist_ctx, dm_stack_glob):
    files = dist_ctx.executor.run_function(lambda: list(sorted(dm_stack_glob)))
    print(files)
    ds = DMDataSet(files=files)
    ds = ds.initialize(dist_ctx.executor)
    analysis = dist_ctx.create_sum_analysis(dataset=ds)
    roi = np.random.choice([True, False], size=len(files))
    results = dist_ctx.run(analysis, roi=roi)
    assert results[0].raw_data.shape == (3838, 3710)


def test_dm_stack_fileset_offsets(dm_stack_of_3d, lt_ctx):
    fs = dm_stack_of_3d._get_fileset()
    f0, f1 = fs

    # check that the offsets in the fileset are correct:
    assert f0.num_frames == 20
    assert f1.num_frames == 20
    assert f0.start_idx == 0
    assert f0.end_idx == 20
    assert f1.start_idx == 20
    assert f1.end_idx == 40

    lt_ctx.run_udf(dataset=dm_stack_of_3d, udf=SumUDF())


@pytest.mark.parametrize(
    "io_backend", (
        BufferedBackend(),
        MMapBackend(),
    ),
)
def test_positive_sync_offset(lt_ctx, io_backend, dm_stack_glob):
    udf = SumSigUDF()
    sync_offset = 2

    ds = lt_ctx.load(
        "dm",
        files=list(sorted(dm_stack_glob)),
        nav_shape=(4, 2),
        io_backend=io_backend,
    )

    result = lt_ctx.run_udf(dataset=ds, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    ds_with_offset = lt_ctx.load(
        "dm",
        files=list(sorted(dm_stack_glob)),
        nav_shape=(4, 2),
        sync_offset=sync_offset,
        io_backend=io_backend,
    )

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
        :ds_with_offset._meta.shape.nav.size - sync_offset
    ]

    assert np.allclose(result, result_with_offset)


@pytest.mark.parametrize(
    "io_backend", (
        BufferedBackend(),
        MMapBackend(),
    ),
)
def test_negative_sync_offset(lt_ctx, io_backend, dm_stack_glob):
    udf = SumSigUDF()
    sync_offset = -2

    ds = lt_ctx.load(
        "dm",
        files=list(sorted(dm_stack_glob)),
        nav_shape=(4, 2),
        io_backend=io_backend,
    )

    result = lt_ctx.run_udf(dataset=ds, udf=udf)
    result = result['intensity'].raw_data[:ds._meta.shape.nav.size - abs(sync_offset)]

    ds_with_offset = lt_ctx.load(
        "dm",
        files=list(sorted(dm_stack_glob)),
        nav_shape=(4, 2),
        sync_offset=sync_offset,
        io_backend=io_backend,
    )

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


@pytest.mark.parametrize(
    "io_backend", (
        BufferedBackend(),
        MMapBackend(),
    ),
)
def test_missing_frames(lt_ctx, io_backend, dm_stack_glob):
    """
    there can be some frames missing at the end
    """
    # one full row of additional frames in the data set than the number of files
    nav_shape = (3, 5)

    ds = lt_ctx.load(
        "dm",
        files=list(sorted(dm_stack_glob)),
        nav_shape=nav_shape,
        io_backend=io_backend,
        num_partitions=4,
    )

    tileshape = Shape(
        (1,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p._start_frame == 11
    assert p._num_frames == 4
    assert p.slice.origin == (11, 0, 0)
    assert p.slice.shape[0] == 4
    assert t.tile_slice.origin == (9, 0, 0)
    assert t.tile_slice.shape[0] == 1


def test_scheme_too_large(default_dm):
    partitions = default_dm.get_partitions()
    p = next(partitions)
    depth = p.shape[0]

    # we make a tileshape that is too large for the partition here:
    tileshape = Shape(
        (depth + 1,) + tuple(default_dm.shape.sig),
        sig_dims=default_dm.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_dm.shape,
    )

    # tile shape is clamped to partition shape:
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == tuple((depth,) + default_dm.shape.sig)
    for _ in tiles:
        pass


def test_offset_smaller_than_image_count(lt_ctx, dm_stack_glob):
    sync_offset = -12

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "dm",
            files=list(sorted(dm_stack_glob)),
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-10, 10\), which is \(-image_count, image_count\)"
    )


def test_offset_greater_than_image_count(lt_ctx, dm_stack_glob):
    sync_offset = 12

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "dm",
            files=list(sorted(dm_stack_glob)),
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-10, 10\), which is \(-image_count, image_count\)"
    )


@pytest.mark.slow
def test_reshape_nav(lt_ctx, dm_stack_glob):
    udf = SumSigUDF()
    files = list(sorted(dm_stack_glob))

    ds_with_1d_nav = lt_ctx.load("dm", files=files, nav_shape=(8,))
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    ds_with_2d_nav = lt_ctx.load("dm", files=files, nav_shape=(4, 2))
    result_with_2d_nav = lt_ctx.run_udf(dataset=ds_with_2d_nav, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load("dm", files=files, nav_shape=(2, 2, 2))
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


def test_incorrect_sig_shape(lt_ctx, dm_stack_glob):
    sig_shape = (5, 5)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "dm",
            files=list(sorted(dm_stack_glob)),
            sig_shape=sig_shape
        )
    assert e.match(
        r"sig_shape must be of size: 14238980"
    )


def test_scan_size_deprecation(lt_ctx, dm_stack_glob):
    scan_size = (2, 2)

    with pytest.warns(FutureWarning):
        ds = lt_ctx.load(
            "dm",
            files=list(sorted(dm_stack_glob)),
            scan_size=scan_size,
        )
    assert tuple(ds.shape) == (2, 2, 3838, 3710)


def test_compare_backends(lt_ctx, default_dm, buffered_dm):
    x = random.choice(range(default_dm.shape.nav[0]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_dm,
        x=x,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=buffered_dm,
        x=x,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="No support for direct I/O on Mac OS X"
)
def test_compare_direct_to_mmap(lt_ctx, default_dm, direct_dm):
    x = random.choice(range(default_dm.shape.nav[0]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_dm,
        x=x,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=direct_dm,
        x=x,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


@pytest.mark.parametrize(
    "as_sparse", (
        False,
        True
    ),
)
def test_compare_backends_sparse(lt_ctx, default_dm, buffered_dm, as_sparse):
    roi = np.zeros(default_dm.shape.nav, dtype=bool).reshape((-1,))
    roi[0] = True
    roi[1] = True
    roi[-1] = True
    if as_sparse:
        roi = roi_as_sparse(roi)
    mm_f0 = lt_ctx.run_udf(dataset=default_dm, udf=PickUDF(), roi=roi)['intensity']
    buffered_f0 = lt_ctx.run_udf(dataset=buffered_dm, udf=PickUDF(), roi=roi)['intensity']

    assert np.allclose(mm_f0, buffered_f0)


def test_nonstack_files(lt_ctx):
    files = '/path/to/dm/file.dm3'

    with pytest.raises(DataSetException):
        lt_ctx.load(
            "dm",
            files=files,
        )


def test_stack_in_an_inline_pickle(default_dm, lt_ctx):
    # make sure the __new__ magic works with pickle
    import cloudpickle
    ds = cloudpickle.loads(cloudpickle.dumps(default_dm))
    lt_ctx.run_udf(dataset=ds, udf=SumUDF())


def test_load_stack_dd(local_cluster_ctx, dm_stack_glob):
    files = dm_stack_glob
    ds = local_cluster_ctx.load("dm", files=files, same_offset=True)
    ds.check_valid()


def test_bad_params(ds_params_tester, standard_bad_ds_params, dm_3d_glob):
    for params in standard_bad_ds_params:
        ds_params_tester("dm", files=dm_3d_glob, **params)
