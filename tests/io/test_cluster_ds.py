import os

import pytest
import numpy as np

from libertem.io.dataset.base import PartitionStructure
from libertem.io.dataset.cluster import ClusterDataSet


@pytest.fixture(scope='function')
def draw_directory(tmpdir_factory):
    datadir = tmpdir_factory.mktemp('draw_directory')
    yield datadir


def test_structure_from_json():
    data = {
        "version": 1,
        "slices": [[0, 127], [128, 256]],
        "shape": [16, 16, 128, 128],
        "sig_dims": 2,
        "dtype": "float32",
    }
    struct = PartitionStructure.from_json(data)
    assert struct.shape.sig.dims == 2
    assert tuple(struct.shape) == (16, 16, 128, 128)
    assert struct.slices == [(0, 127), (128, 256)]
    assert struct.dtype == np.dtype("float32")


def test_initialization(draw_directory, lt_ctx):
    data = {
        "version": 1,
        "slices": [[0, 8], [16, 256]],
        "shape": [4, 4, 128, 128],
        "sig_dims": 2,
        "dtype": "float32",
    }
    structure = PartitionStructure.from_json(data)
    ds = ClusterDataSet(path=draw_directory, enable_direct=False, structure=structure)
    ds = ds.initialize(lt_ctx.executor)
    ds.check_valid()

    assert os.listdir(draw_directory) == ["structure.json", "parts"]


def test_inconsistent_sidecars_raise_error():
    # TODO: need multiple "hosts" for this to work
    # maybe create a simple InlineExecutor-derived executor for this?
    # TODO: need path-per-host for this; maybe `path` can be a dict? or make a second arg paths...
    pass


def test_missing_sidecar_are_created():
    # TODO: see above; also need multiple hosts
    pass
