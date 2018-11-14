import os
import tempfile

import pytest
import h5py
import numpy as np
from scipy.ndimage import measurements

from libertem.io.dataset.hdf5 import H5DataSet
from libertem.executor.inline import InlineJobExecutor
from libertem.analysis.com import COMAnalysis


@pytest.fixture
def hdf5():
    f, tmpfn = tempfile.mkstemp(suffix=".h5")
    os.close(f)
    with h5py.File(tmpfn, "w") as f:
        yield f
    os.unlink(tmpfn)


@pytest.fixture
def hdf5_ds_1(hdf5):
    hdf5.create_dataset("data", data=np.ones((5, 5, 16, 16)))
    return hdf5


@pytest.fixture
def hdf5_ds_2(hdf5):
    hdf5.create_dataset("data", data=np.zeros((5, 5, 16, 16)))
    return hdf5


@pytest.fixture
def hdf5_ds_3(hdf5):
    hdf5.create_dataset("data", data=np.ones((5, 5, 3, 3)))
    return hdf5


def do_com(fn, tileshape):
    ds = H5DataSet(
        path=fn,
        ds_path="data",
        tileshape=tileshape,
        target_size=512*1024*1024,
    )

    params = {
        "cx": 0,
        "cy": 0,
        "r": 2 * max(ds.shape),
    }

    analysis = COMAnalysis(dataset=ds, parameters=params)
    job = analysis.get_job()
    print(job.masks.computed_masks)
    print("\n\n")
    executor = InlineJobExecutor()
    full_result = job.get_result_buffer()
    # color = np.zeros(shape=(3,) + ds.shape[:2])
    for result in executor.run_job(job):
        for tile in result:
            print(tile)
            print(tile.data[0])
            # color[tile.tile_slice.get()[:2]] += 1
            tile.copy_to_result(full_result)
    # print(color)

    results = analysis.get_results(full_result)
    x_centers = results[3].raw_data
    y_centers = results[4].raw_data

    return full_result, x_centers, y_centers


def test_center_of_mass_from_h5_1(hdf5_ds_1):
    full, x, y = do_com(hdf5_ds_1.filename, tileshape=(1, 3, 6, 6))
    assert x.shape == (5, 5)
    assert y.shape == (5, 5)
    assert full.shape == (3, 5, 5)
    assert full[0][0, 0] == hdf5_ds_1['data'][0, 0].sum()
    scx, scy = measurements.center_of_mass(hdf5_ds_1['data'][0, 0])
    print(full[0])
    print(full[1])
    print(full[2])
    assert scx == 7.5
    assert np.all(x == 7.5)
    assert np.all(y == 7.5)
    assert np.all(full[0] == 16 * 16)               # sum over intensities (=1) of 16x16 input
    assert np.all(full[1] == sum(range(16)) * 16)   # sum over gradients from 0 to 15 in each row
    assert np.all(full[2] == sum(range(16)) * 16)   # sum over gradients from 0 to 15 in each col


def test_center_of_mass_from_h5_1_1(hdf5_ds_1):
    full, x, y = do_com(hdf5_ds_1.filename, tileshape=(1, 1, 16, 16))
    assert x.shape == (5, 5)
    assert y.shape == (5, 5)
    assert full.shape == (3, 5, 5)
    scx, scy = measurements.center_of_mass(hdf5_ds_1['data'][0, 0])
    assert scx == 7.5
    assert np.all(x == 7.5)
    assert np.all(y == 7.5)
    assert np.all(full[0] == 16 * 16)               # sum over intensities (=1) of 16x16 input
    assert np.all(full[1] == sum(range(16)) * 16)   # sum over gradients from 0 to 15 in each row
    assert np.all(full[2] == sum(range(16)) * 16)   # sum over gradients from 0 to 15 in each col
    print(full[0])
    print(full[1])
    print(full[2])


def test_center_of_mass_from_h5_1_2(hdf5_ds_1):
    full, x, y = do_com(hdf5_ds_1.filename, tileshape=(1, 5, 16, 16))
    assert x.shape == (5, 5)
    assert y.shape == (5, 5)
    assert full.shape == (3, 5, 5)
    scx, scy = measurements.center_of_mass(hdf5_ds_1['data'][0, 0])
    assert scx == 7.5
    assert np.all(x == 7.5)
    assert np.all(y == 7.5)
    assert np.all(full[0] == 16 * 16)               # sum over intensities (=1) of 16x16 input
    assert np.all(full[1] == sum(range(16)) * 16)   # sum over gradients from 0 to 15 in each row
    assert np.all(full[2] == sum(range(16)) * 16)   # sum over gradients from 0 to 15 in each col


def test_center_of_mass_from_h5_2(hdf5_ds_2):
    full, x, y = do_com(hdf5_ds_2.filename, tileshape=(1, 3, 6, 6))
    assert x.shape == (5, 5)
    assert y.shape == (5, 5)
    assert full.shape == (3, 5, 5)
    assert not np.any(np.isnan(x))
    assert not np.any(np.isnan(y))


def test_center_of_mass_from_h5_3(hdf5_ds_3):
    """
    ds shape: (5, 5, 3, 3)
    5x5 scan positions
    3x3 pixels per frame
    data: all ones
    """
    full, x, y = do_com(hdf5_ds_3.filename, tileshape=(1, 3, 2, 2))
    assert x.shape == (5, 5)
    assert y.shape == (5, 5)
    assert full.shape == (3, 5, 5)
    assert np.all(full[0] == 3 * 3)
    assert np.all(full[1] == sum(range(3)) * 3)
    assert np.all(full[2] == sum(range(3)) * 3)
    assert np.all(x == 1)
    assert np.all(y == 1)

    scx, scy = measurements.center_of_mass(hdf5_ds_3['data'][0, 0])
    assert scx == 1
    assert scy == 1
