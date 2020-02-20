import os

import pytest
import numpy as np

from libertem.common.slice import Slice, Shape
from libertem.io.dataset.base import DataTile
from libertem.io.writers.base import WriteHandle


def test_write_handle(tmpdir_factory):
    """
    test the common "happy path":
    """
    datadir = tmpdir_factory.mktemp('write_handle_tests')
    full_path = os.path.join(datadir, "f1")
    part_slice = Slice(
        shape=Shape((32, 64, 64), sig_dims=2),
        origin=(16, 0, 0),
    )
    tile_slice = Slice(
        shape=Shape((3, 64, 64), sig_dims=2),
        origin=(19, 0, 0),
    )
    tile_data = np.random.randn(3, 64, 64).astype("float32")
    tile = DataTile(
        tile_data,
        tile_slice=tile_slice,
        scheme_idx=0,
    )

    wh = WriteHandle(full_path, datadir, part_slice, dtype='float32')

    tmp_file_name = ""

    with wh:
        wh.write_tile(tile)
        tmp_file_name = wh._tmp_file.name
        assert os.path.exists(tmp_file_name)

    # check some internals:
    assert wh._dest is None
    assert wh._tmp_file is None

    # the temporary file should no longer exist in case of success
    assert not os.path.exists(tmp_file_name)

    # ... buf our dest fname should:
    assert os.path.exists(full_path)
    assert os.path.isfile(full_path)

    # check if data is written correctly:
    read_data = np.fromfile(full_path, dtype="float32").reshape(part_slice.shape)
    assert np.allclose(
        read_data[3:6, ...],
        tile_data
    )


def test_write_handle_aborted(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('write_handle_tests')
    full_path = os.path.join(datadir, "f1")
    part_slice = Slice(
        shape=Shape((32, 64, 64), sig_dims=2),
        origin=(16, 0, 0),
    )
    tile_slice = Slice(
        shape=Shape((3, 64, 64), sig_dims=2),
        origin=(19, 0, 0),
    )
    tile_data = np.random.randn(3, 64, 64).astype("float32")
    tile = DataTile(
        tile_data,
        tile_slice=tile_slice,
        scheme_idx=0,
    )

    wh = WriteHandle(full_path, datadir, part_slice, dtype='float32')

    tmp_file_name = ""

    with wh:
        wh.write_tile(tile)
        tmp_file_name = wh._tmp_file.name
        assert os.path.exists(tmp_file_name)
        wh.abort()
        assert not os.path.exists(tmp_file_name)

    # check some internals:
    assert wh._dest is None
    assert wh._tmp_file is None

    # the temporary file should no longer exist in case of abortion
    assert not os.path.exists(tmp_file_name)

    # and neither should the full destination path
    assert not os.path.exists(full_path)


def test_write_handle_exception(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('write_handle_tests')
    full_path = os.path.join(datadir, "f1")
    part_slice = Slice(
        shape=Shape((32, 64, 64), sig_dims=2),
        origin=(16, 0, 0),
    )
    tile_slice = Slice(
        shape=Shape((3, 64, 64), sig_dims=2),
        origin=(19, 0, 0),
    )
    tile_data = np.random.randn(3, 64, 64).astype("float32")
    tile = DataTile(
        tile_data,
        tile_slice=tile_slice,
        scheme_idx=0,
    )

    wh = WriteHandle(full_path, datadir, part_slice, dtype='float32')

    tmp_file_name = ""

    with pytest.raises(Exception):
        with wh:
            wh.write_tile(tile)
            tmp_file_name = wh._tmp_file.name
            assert os.path.exists(tmp_file_name)
            raise Exception("nope")

    # check some internals:
    assert wh._dest is None
    assert wh._tmp_file is None

    # the temporary file should no longer exist in case of exception
    assert not os.path.exists(tmp_file_name)

    # and neither should the full destination path
    assert not os.path.exists(full_path)
