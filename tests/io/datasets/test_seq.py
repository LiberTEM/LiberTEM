import os
import sys
import random

import numpy as np
import pytest

from libertem.executor.inline import InlineJobExecutor
from libertem.io.dataset.seq import (SEQDataSet, _load_xml_from_string, xml_defect_data_extractor,
                                     xml_map_sizes, xml_unbinned_map_maker, array_cropping,
                                     xml_defect_coord_extractor, xml_map_index_selector,
                                     xml_binned_map_maker)
from libertem.common import Shape
from libertem.common.buffers import reshaped_view
from libertem.udf.sumsigudf import SumSigUDF
from libertem.udf.raw import PickUDF
from libertem.io.dataset.base import TilingScheme, BufferedBackend, MMapBackend, DirectBackend
from libertem.io.corrections import CorrectionSet

from utils import get_testdata_path, ValidationUDF, roi_as_sparse
import defusedxml.ElementTree as ET

try:
    # FIXME: pims/skimage can't deal with numpy2 on python 3.9
    if int(np.version.version.split('.')[0]) < 2 or sys.version_info >= (3, 10):
        import pims
    else:
        pims = None
except ModuleNotFoundError:
    pims = None

SEQ_TESTDATA_PATH = os.path.join(get_testdata_path(), 'default.seq')
HAVE_SEQ_TESTDATA = os.path.exists(SEQ_TESTDATA_PATH)

needsdata = pytest.mark.skipif(not HAVE_SEQ_TESTDATA, reason="need .seq testdata")


@pytest.fixture
def default_seq(lt_ctx):
    nav_shape = (8, 8)
    ds = lt_ctx.load(
        "seq",
        path=SEQ_TESTDATA_PATH,
        nav_shape=nav_shape,
        io_backend=MMapBackend(),
        num_partitions=4,
    )

    assert tuple(ds.shape) == (8, 8, 128, 128)
    return ds


@pytest.fixture
def buffered_seq(lt_ctx):
    nav_shape = (8, 8)

    ds = lt_ctx.load(
        "seq",
        path=SEQ_TESTDATA_PATH,
        nav_shape=nav_shape,
        io_backend=BufferedBackend(),
        num_partitions=4,
    )

    return ds


@pytest.fixture
def direct_seq(lt_ctx):
    nav_shape = (8, 8)

    ds = lt_ctx.load(
        "seq",
        path=SEQ_TESTDATA_PATH,
        nav_shape=nav_shape,
        io_backend=DirectBackend(),
        num_partitions=4,
    )

    return ds


@pytest.fixture(scope='module')
def default_seq_raw():
    return np.array(pims.open(str(SEQ_TESTDATA_PATH))).reshape((8, 8, 128, 128))


@pytest.mark.skipif(pims is None, reason="No PIMS found")
@needsdata
def test_comparison(default_seq, default_seq_raw, lt_ctx_fast):
    corrset = CorrectionSet()
    udf = ValidationUDF(
        reference=reshaped_view(default_seq_raw, (-1, *tuple(default_seq.shape.sig)))
    )
    lt_ctx_fast.run_udf(udf=udf, dataset=default_seq, corrections=corrset)


@pytest.mark.skipif(pims is None, reason="No PIMS found")
@needsdata
def test_comparison_roi(default_seq, default_seq_raw, lt_ctx_fast):
    corrset = CorrectionSet()
    roi = np.random.choice(
        [True, False],
        size=tuple(default_seq.shape.nav),
        p=[0.5, 0.5]
    )
    udf = ValidationUDF(reference=default_seq_raw[roi])
    lt_ctx_fast.run_udf(udf=udf, dataset=default_seq, roi=roi, corrections=corrset)


@needsdata
def test_positive_sync_offset(default_seq, lt_ctx):
    udf = SumSigUDF()
    sync_offset = 2

    ds_with_offset = SEQDataSet(
        path=SEQ_TESTDATA_PATH, nav_shape=(8, 8), sync_offset=sync_offset,
        num_partitions=4,
    )
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == 2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (4,) + tuple(ds_with_offset.shape.sig),
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

    assert p.slice.origin == (48, 0, 0)
    assert p.slice.shape[0] == 16

    result = lt_ctx.run_udf(dataset=default_seq, udf=udf)
    result = result['intensity'].raw_data[sync_offset:]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[
                         :ds_with_offset._meta.image_count - sync_offset
                         ]

    assert np.allclose(result, result_with_offset)


def test_array_cropping():
    start_size = (1024, 1024)
    crop_to_this = (512, 512)
    offset = (600, 600)
    array = np.zeros(start_size)
    n_array = array_cropping(array, start_size, crop_to_this, offset)
    assert np.array_equal(n_array, array)


def test_xml_excluded_pixels_unbinned():
    xml_string = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                    <Configuration>
                        <PixelSize></PixelSize><DiffPixelSize></DiffPixelSize>
                        <BadPixels>
                        <BadPixelMap Rows="4096" Columns="4096">
                            <Defect Rows="2311-2312"/>
                            <Defect Rows="3413-3414"/>
                            <Defect Column="2311"/>
                        </BadPixelMap>
                        <BadPixelMap Binning="2" Rows="2048" Columns="2048">
                            <Defect Rows="1155-1156"/>
                            <Defect Rows="1706-1707"/>
                        </BadPixelMap>
                        </BadPixels>
                    </Configuration>
            '''
    metadata = {
        "UnbinnedFrameSizeX": 1024,
        "UnbinnedFrameSizeY": 1024,
        "OffsetX": 1536,
        "OffsetY": 1536,
        "HardwareBinning": 1
    }
    test_arr = np.zeros((1024, 1024), dtype=bool)
    test_arr[775] = True
    test_arr[:, 775] = True
    test_arr[776] = True
    expected_res = _load_xml_from_string(xml=xml_string, metadata=metadata)
    assert np.array_equal(expected_res.todense(), test_arr)


def test_xml_excluded_pixels_only_binned():
    xml_string = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                    <Configuration>
                        <PixelSize></PixelSize><DiffPixelSize></DiffPixelSize>
                        <BadPixels>
                            <BadPixelMap Rows="4096" Columns="4096">
                                <Defect Rows="2311-2312"/>
                                <Defect Row="600"/>
                                <Defect Columns="1310-1312"/>
                                <Defect Column="1300"/>
                                <Defect Row="100" Column="150"/>
                            </BadPixelMap>
                            <BadPixelMap Binning="2" Rows="2048" Columns="2048">
                                <Defect Rows="1155-1156"/>
                                <Defect Row="300"/>
                                <Defect Columns="655-656"/>
                                <Defect Column="650"/>
                                <Defect Row="50" Column="75"/>
                            </BadPixelMap>
                        </BadPixels>
                    </Configuration>
        '''
    metadata = {
        "UnbinnedFrameSizeX": 4096,
        "UnbinnedFrameSizeY": 4096,
        "OffsetX": 0,
        "OffsetY": 0,
        "HardwareBinning": 2
    }
    test_arr = np.zeros((2048, 2048), dtype=bool)
    test_arr[1155] = True
    test_arr[1156] = True
    test_arr[300] = True
    test_arr[50, 75] = True
    test_arr[:, 650] = True
    test_arr[:, 655:657] = True
    expected_res = _load_xml_from_string(xml=xml_string, metadata=metadata)
    assert np.array_equal(expected_res.todense(), test_arr)


def test_xml_excluded_pixels_binned_cropped():
    xml_string = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                    <Configuration>
                        <PixelSize></PixelSize><DiffPixelSize></DiffPixelSize>
                        <BadPixels>
                            <BadPixelMap Rows="4096" Columns="4096">
                                <Defect Rows="2311-2312"/>
                                <Defect Rows="3413-3414"/>
                                <Defect Columns="1310-1312"/>
                                <Defect Column="1300"/>
                                <Defect Row="100" Column="150"/>
                            </BadPixelMap>
                            <BadPixelMap Binning="2" Rows="2048" Columns="2048">
                                <Defect Rows="1250-1252"/>
                                <Defect Row="800"/>
                                <Defect Columns="768-770"/>
                                <Defect Column="1000"/>
                                <Defect Row="1200" Column="1100"/>
                            </BadPixelMap>
                        </BadPixels>
                    </Configuration>
        '''
    metadata = {
        "UnbinnedFrameSizeX": 1024,
        "UnbinnedFrameSizeY": 1024,
        "OffsetX": 1536,
        "OffsetY": 1536,
        "HardwareBinning": 2
    }
    test_arr = np.zeros((512, 512), dtype=bool)
    test_arr[482:485] = True
    test_arr[32] = True
    test_arr[:, 0:3] = True
    test_arr[:, 232] = True
    test_arr[432, 332] = True
    expected_res = _load_xml_from_string(xml=xml_string, metadata=metadata)
    assert np.array_equal(expected_res.todense(), test_arr)


def test_correct_bad_pixel_map_selector():
    xml_string = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                    <Configuration>
                        <PixelSize></PixelSize><DiffPixelSize></DiffPixelSize>
                        <BadPixels>
                            <BadPixelMap Rows="4096" Columns="4096">
                                <Defect Columns="1310-1312"/>
                            </BadPixelMap>
                            <BadPixelMap Rows="2048" Columns="2048">
                                <Defect Rows="1155-1156"/>
                                <Defect Rows="1706-1707"/>
                            </BadPixelMap>
                        </BadPixels>
                    </Configuration>
        '''
    metadata = {
        "UnbinnedFrameSizeX": 1024,
        "UnbinnedFrameSizeY": 1024,
        "OffsetX": 1536,
        "OffsetY": 1536,
        "HardwareBinning": 1
    }
    tree = ET.fromstring(xml_string)
    excluded_rows_dict = xml_defect_data_extractor(tree, metadata)
    assert excluded_rows_dict["cols"] == [['1310', '1312']]


def test_correct_bad_pixel_map_selector_2():
    xml_string = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
                    <Configuration>
                        <PixelSize></PixelSize><DiffPixelSize></DiffPixelSize>
                        <BadPixels>
                            <BadPixelMap Rows="4096" Columns="4096">
                                <Defect Columns="1310-1312"/>
                            </BadPixelMap>
                            <BadPixelMap Binning="2" Rows="2048" Columns="2048">
                                <Defect Rows="1155-1156"/>
                            </BadPixelMap>
                        </BadPixels>
                    </Configuration>
        '''
    metadata = {
        "UnbinnedFrameSizeX": 1024,
        "UnbinnedFrameSizeY": 1024,
        "OffsetX": 1536,
        "OffsetY": 1536,
        "HardwareBinning": 2
    }
    tree = ET.fromstring(xml_string)
    excluded_rows_dict = xml_defect_data_extractor(tree, metadata)
    assert excluded_rows_dict["rows"] == [['1155', '1156']]


def test_map_size():
    xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
          <BadPixels>
              <BadPixelMap Rows="4096" Columns="4096"></BadPixelMap>
              <BadPixelMap Binning="2" Rows="4080" Columns="4096"></BadPixelMap>
          </BadPixels>
          '''
    tree = ET.fromstring(xml)
    bad_pixel_maps = tree.findall('.//BadPixelMap')
    xy_map_sizes, map_sizes = xml_map_sizes(bad_pixel_maps)
    xy_map_sizes_expected = [(4096, 4096), (4096, 4080), (1, 2)]
    assert xy_map_sizes == xy_map_sizes_expected


def test_unbinned_map_maker():
    xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
          <BadPixels>
              <BadPixelMap Rows="4096" Columns="4096"></BadPixelMap>
              <BadPixelMap Rows="2048" Columns="2048"></BadPixelMap>
              <BadPixelMap Binning="2" Rows="4080" Columns="4096"></BadPixelMap>
          </BadPixels>
         '''
    tree = ET.fromstring(xml)
    bad_pixel_maps = tree.findall('.//BadPixelMap')
    xy_map_sizes, _ = xml_map_sizes(bad_pixel_maps)
    unbinned_x, unbinned_y = xml_unbinned_map_maker(xy_map_sizes)
    assert unbinned_x == [4096, 2048, 0]
    assert unbinned_y == [4096, 2048, 0]


def test_binned_map_maker():
    xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
          <BadPixels>
              <BadPixelMap Rows="4096" Columns="4096"></BadPixelMap>
              <BadPixelMap Rows="2048" Columns="2048"></BadPixelMap>
              <BadPixelMap Binning="2" Rows="4096" Columns="4096"></BadPixelMap>
          </BadPixels>
         '''
    tree = ET.fromstring(xml)
    bad_pixel_maps = tree.findall('.//BadPixelMap')
    xy_map_sizes, _ = xml_map_sizes(bad_pixel_maps)
    unbinned_x, unbinned_y = xml_binned_map_maker(xy_map_sizes)
    assert unbinned_x == [0, 0, 4096]
    assert unbinned_y == [0, 0, 4096]


def test_map_index_selector_case1():
    xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
          <BadPixels>
              <BadPixelMap Rows="4096" Columns="4096"></BadPixelMap>
              <BadPixelMap Rows="2048" Columns="2048"></BadPixelMap>
              <BadPixelMap Binning="2" Rows="512" Columns="512"></BadPixelMap>
          </BadPixels>
         '''
    tree = ET.fromstring(xml)
    bad_pixel_maps = tree.findall('.//BadPixelMap')
    xy_map_sizes, map_sizes = xml_map_sizes(bad_pixel_maps)
    used_x, used_y = xml_unbinned_map_maker(xy_map_sizes)
    map_index = xml_map_index_selector(used_y)
    expected_index = 0
    assert map_index == expected_index


def test_map_index_selector_case2():
    xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
          <BadPixels>
              <BadPixelMap Rows="4096" Columns="4096"></BadPixelMap>
              <BadPixelMap Rows="2048" Columns="2048"></BadPixelMap>
              <BadPixelMap Binning="2" Rows="512" Columns="512"></BadPixelMap>
          </BadPixels>
         '''
    tree = ET.fromstring(xml)
    bad_pixel_maps = tree.findall('.//BadPixelMap')
    xy_map_sizes, map_sizes = xml_map_sizes(bad_pixel_maps)
    used_x, used_y = xml_binned_map_maker(xy_map_sizes)
    map_index = xml_map_index_selector(used_y)
    print(used_y)
    expected_index = 2

    assert map_index == expected_index


def test_defect_extractor():
    xml = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
          <BadPixels>
              <BadPixelMap Rows="4096" Columns="4096">
                  <Defect Rows="2311-2312" />
                  <Defect Row="1300" />
                  <Defect Columns="2300-2301" />
                  <Defect Column="600" />
                  <Defect Row="230" Column="100" />
              </BadPixelMap>
              <BadPixelMap Rows="2048" Columns="2048">
                    <Defect Column="10"/>
              </BadPixelMap>
              <BadPixelMap Binning="2" Rows="4080" Columns="4096"></BadPixelMap>
          </BadPixels>
          '''
    tree = ET.fromstring(xml)
    bad_pixel_maps = tree.findall('.//BadPixelMap')
    xy_map_sizes, map_sizes = xml_map_sizes(bad_pixel_maps)
    used_x, used_y = xml_unbinned_map_maker(xy_map_sizes)
    map_index = xml_map_index_selector(used_y)
    defects = xml_defect_coord_extractor(bad_pixel_maps[map_index], map_index, map_sizes)
    expected_defects = {"rows": [['2311', '2312'], ['1300']],
                        "cols": [['2300', '2301'], ['600']],
                        "pixels": [['100', '230']],
                        "size": (4096, 4096)
                        }
    assert defects == expected_defects


@needsdata
def test_negative_sync_offset(default_seq, lt_ctx):
    udf = SumSigUDF()
    sync_offset = -2

    ds_with_offset = SEQDataSet(
        path=SEQ_TESTDATA_PATH, nav_shape=(8, 8), sync_offset=sync_offset,
        num_partitions=4,
    )
    ds_with_offset = ds_with_offset.initialize(lt_ctx.executor)
    ds_with_offset.check_valid()

    p0 = next(ds_with_offset.get_partitions())
    assert p0._start_frame == -2
    assert p0.slice.origin == (0, 0, 0)

    tileshape = Shape(
        (4,) + tuple(ds_with_offset.shape.sig),
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

    assert p.slice.origin == (48, 0, 0)
    assert p.slice.shape[0] == 16

    result = lt_ctx.run_udf(dataset=default_seq, udf=udf)
    result = result['intensity'].raw_data[:default_seq._meta.image_count - abs(sync_offset)]

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf)
    result_with_offset = result_with_offset['intensity'].raw_data[abs(sync_offset):]

    assert np.allclose(result, result_with_offset)


@needsdata
def test_missing_frames(lt_ctx):
    nav_shape = (16, 8)

    ds = SEQDataSet(
        path=SEQ_TESTDATA_PATH,
        nav_shape=nav_shape,
        num_partitions=4,
    )
    ds = ds.initialize(lt_ctx.executor)
    ds.check_valid()

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p._start_frame == 96
    assert p._num_frames == 32
    assert p.slice.origin == (96, 0, 0)
    assert p.slice.shape[0] == 32
    assert t.tile_slice.origin == (60, 0, 0)
    assert t.tile_slice.shape[0] == 4


@needsdata
def test_missing_data_with_positive_sync_offset(lt_ctx):
    nav_shape = (16, 8)
    sync_offset = 8

    ds = SEQDataSet(
        path=SEQ_TESTDATA_PATH,
        nav_shape=nav_shape,
        sync_offset=sync_offset,
        num_partitions=4,
    )
    ds = ds.initialize(lt_ctx.executor)

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p._start_frame == 104
    assert p._num_frames == 32
    assert t.tile_slice.origin == (52, 0, 0)
    assert t.tile_slice.shape[0] == 4


@needsdata
def test_missing_data_with_negative_sync_offset(lt_ctx):
    nav_shape = (16, 8)
    sync_offset = -8

    ds = SEQDataSet(
        path=SEQ_TESTDATA_PATH,
        nav_shape=nav_shape,
        sync_offset=sync_offset,
        num_partitions=4,
    )
    ds = ds.initialize(lt_ctx.executor)

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass

    assert p._start_frame == 88
    assert p._num_frames == 32
    assert t.tile_slice.origin == (68, 0, 0)
    assert t.tile_slice.shape[0] == 4


@needsdata
def test_too_many_frames(lt_ctx):
    nav_shape = (4, 8)

    ds = SEQDataSet(
        path=SEQ_TESTDATA_PATH,
        nav_shape=nav_shape,
        num_partitions=4,
    )
    ds = ds.initialize(lt_ctx.executor)
    ds.check_valid()

    tileshape = Shape(
        (4,) + tuple(ds.shape.sig),
        sig_dims=ds.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=ds.shape,
    )

    for p in ds.get_partitions():
        for t in p.get_tiles(tiling_scheme=tiling_scheme):
            pass


@needsdata
def test_positive_sync_offset_with_roi(default_seq, lt_ctx):
    udf = SumSigUDF()
    result = lt_ctx.run_udf(dataset=default_seq, udf=udf)
    result = result['intensity'].raw_data

    nav_shape = (8, 8)
    sync_offset = 2

    ds_with_offset = lt_ctx.load(
        "seq", path=SEQ_TESTDATA_PATH, nav_shape=nav_shape, sync_offset=sync_offset
    )

    roi = np.random.choice([False], (8, 8))
    roi[0:1] = True

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
    result_with_offset = result_with_offset['intensity'].raw_data

    assert np.allclose(result[sync_offset:8 + sync_offset], result_with_offset)


@needsdata
def test_negative_sync_offset_with_roi(default_seq, lt_ctx):
    udf = SumSigUDF()
    result = lt_ctx.run_udf(dataset=default_seq, udf=udf)
    result = result['intensity'].raw_data

    nav_shape = (8, 8)
    sync_offset = -2

    ds_with_offset = lt_ctx.load(
        "seq", path=SEQ_TESTDATA_PATH, nav_shape=nav_shape, sync_offset=sync_offset
    )

    roi = np.random.choice([False], (8, 8))
    roi[0:1] = True

    result_with_offset = lt_ctx.run_udf(dataset=ds_with_offset, udf=udf, roi=roi)
    result_with_offset = result_with_offset['intensity'].raw_data

    assert np.allclose(result[:8 + sync_offset], result_with_offset[abs(sync_offset):])


@needsdata
def test_offset_smaller_than_image_count(lt_ctx):
    nav_shape = (8, 8)
    sync_offset = -65

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "seq",
            path=SEQ_TESTDATA_PATH,
            nav_shape=nav_shape,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-64, 64\), which is \(-image_count, image_count\)"
    )


@needsdata
def test_offset_greater_than_image_count(lt_ctx):
    nav_shape = (8, 8)
    sync_offset = 65

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "seq",
            path=SEQ_TESTDATA_PATH,
            nav_shape=nav_shape,
            sync_offset=sync_offset
        )
    assert e.match(
        r"offset should be in \(-64, 64\), which is \(-image_count, image_count\)"
    )


@needsdata
def test_reshape_nav(lt_ctx, default_seq):
    udf = SumSigUDF()

    ds_with_1d_nav = lt_ctx.load("seq", path=SEQ_TESTDATA_PATH, nav_shape=(64,))
    result_with_1d_nav = lt_ctx.run_udf(dataset=ds_with_1d_nav, udf=udf)
    result_with_1d_nav = result_with_1d_nav['intensity'].raw_data

    result_with_2d_nav = lt_ctx.run_udf(dataset=default_seq, udf=udf)
    result_with_2d_nav = result_with_2d_nav['intensity'].raw_data

    ds_with_3d_nav = lt_ctx.load("seq", path=SEQ_TESTDATA_PATH, nav_shape=(2, 4, 8))
    result_with_3d_nav = lt_ctx.run_udf(dataset=ds_with_3d_nav, udf=udf)
    result_with_3d_nav = result_with_3d_nav['intensity'].raw_data

    assert np.allclose(result_with_1d_nav, result_with_2d_nav, result_with_3d_nav)


@needsdata
def test_reshape_different_shapes(lt_ctx, default_seq):
    udf = SumSigUDF()

    result = lt_ctx.run_udf(dataset=default_seq, udf=udf)
    result = result['intensity'].raw_data

    ds_1 = lt_ctx.load("seq", path=SEQ_TESTDATA_PATH, nav_shape=(3, 6))
    result_1 = lt_ctx.run_udf(dataset=ds_1, udf=udf)
    result_1 = result_1['intensity'].raw_data

    assert np.allclose(result_1, result[:3 * 6])


@needsdata
def test_incorrect_sig_shape(lt_ctx):
    nav_shape = (8, 8)
    sig_shape = (5, 5)

    with pytest.raises(Exception) as e:
        lt_ctx.load(
            "seq",
            path=SEQ_TESTDATA_PATH,
            nav_shape=nav_shape,
            sig_shape=sig_shape
        )
    assert e.match(
        r"sig_shape must be of size: 16384"
    )


@needsdata
def test_scan_size_deprecation(lt_ctx):
    scan_size = (5, 5)

    with pytest.warns(FutureWarning):
        ds = lt_ctx.load(
            "seq",
            path=SEQ_TESTDATA_PATH,
            scan_size=scan_size,
        )
    assert tuple(ds.shape) == (5, 5, 128, 128)


@needsdata
def test_detect_non_seq(raw_with_zeros, lt_ctx):
    path = raw_with_zeros._path
    # raw_with_zeros is not a SEQ file, caused UnicodeDecodeError before:
    assert SEQDataSet.detect_params(path, InlineJobExecutor()) is False


@needsdata
def test_detect_seq(lt_ctx):
    path = SEQ_TESTDATA_PATH
    assert SEQDataSet.detect_params(path, lt_ctx.executor) is not False


# from utils import dataset_correction_verification
# FIXME test with actual test file

@needsdata
def test_compare_backends(lt_ctx, default_seq, buffered_seq):
    y = random.choice(range(default_seq.shape.nav[0]))
    x = random.choice(range(default_seq.shape.nav[1]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_seq,
        x=x, y=y,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=buffered_seq,
        x=x, y=y,
    )).intensity

    assert np.allclose(mm_f0, buffered_f0)


@needsdata
@pytest.mark.skipif(
    sys.platform.startswith("darwin"),
    reason="No support for direct I/O on Mac OS X"
)
def test_compare_direct_to_mmap(lt_ctx, default_seq, direct_seq):
    y = random.choice(range(default_seq.shape.nav[0]))
    x = random.choice(range(default_seq.shape.nav[1]))
    mm_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=default_seq,
        x=x, y=y,
    )).intensity
    buffered_f0 = lt_ctx.run(lt_ctx.create_pick_analysis(
        dataset=direct_seq,
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
def test_compare_backends_sparse(lt_ctx, default_seq, buffered_seq, as_sparse):
    roi = np.zeros(default_seq.shape.nav, dtype=bool).reshape((-1,))
    roi[0] = True
    roi[1] = True
    roi[16] = True
    roi[32] = True
    roi[-1] = True
    if as_sparse:
        roi = roi_as_sparse(roi)
    mm_f0 = lt_ctx.run_udf(dataset=default_seq, udf=PickUDF(), roi=roi)['intensity']
    buffered_f0 = lt_ctx.run_udf(dataset=buffered_seq, udf=PickUDF(), roi=roi)['intensity']

    assert np.allclose(mm_f0, buffered_f0)


@needsdata
def test_scheme_too_large(default_seq):
    partitions = default_seq.get_partitions()
    p = next(partitions)
    depth = p.shape[0]

    # we make a tileshape that is too large for the partition here:
    tileshape = Shape(
        (depth + 1,) + tuple(default_seq.shape.sig),
        sig_dims=default_seq.shape.sig.dims
    )
    tiling_scheme = TilingScheme.make_for_shape(
        tileshape=tileshape,
        dataset_shape=default_seq.shape,
    )

    # tile shape is clamped to partition shape:
    tiles = p.get_tiles(tiling_scheme=tiling_scheme)
    t = next(tiles)
    assert tuple(t.tile_slice.shape) == tuple((depth,) + default_seq.shape.sig)
    for _ in tiles:
        pass


def test_bad_params(ds_params_tester, standard_bad_ds_params):
    args = ("seq", SEQ_TESTDATA_PATH)
    for params in standard_bad_ds_params:
        if "nav_shape" not in params:
            params['nav_shape'] = (8, 8)
        ds_params_tester(*args, **params)


@needsdata
def test_no_num_partitions(lt_ctx):
    nav_shape = (8, 8)
    ds = lt_ctx.load(
        "seq",
        path=SEQ_TESTDATA_PATH,
        nav_shape=nav_shape,
    )
    lt_ctx.run_udf(dataset=ds, udf=SumSigUDF())
