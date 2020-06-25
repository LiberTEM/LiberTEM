import os
import json

import pytest
import numpy as np

from libertem.io.dataset.base import PartitionStructure, DataSetException
from libertem.io.dataset.cluster import ClusterDataSet

# FIXME test on actual data structure, including correction
# from utils import dataset_correction_verification


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
    ds = ClusterDataSet(path=str(draw_directory), structure=structure)
    ds = ds.initialize(lt_ctx.executor)
    ds.check_valid()

    assert set(os.listdir(draw_directory)) == {"parts", "structure.json"}


@pytest.mark.dist
def test_inconsistent_sidecars_raise_error(dist_ctx, draw_directory):
    # tmpdir_factory doesn't output plain strings,
    # and thus draw_directory was some pytest object. we don't have
    # pytest installed on the worker containers, so it failed spectacularly:
    draw_directory = str(draw_directory)
    # first, create consistent ClusterDataSet:
    data = {
        "version": 1,
        "slices": [[0, 4], [5, 16]],
        "shape": [4, 4, 128, 128],
        "sig_dims": 2,
        "dtype": "float32",
    }
    structure = PartitionStructure.from_json(data)
    ds = ClusterDataSet(path=draw_directory, structure=structure)
    ds = ds.initialize(dist_ctx.executor)
    ds.check_valid()

    # then, scramble the sidecars, creating an inconsistent state:
    data = {
        "worker-1": {
            "version": 1,
            "slices": [[0, 8], [9, 16]],
            "shape": [4, 4, 128, 128],
            "sig_dims": 2,
            "dtype": "int32",
        },
        "worker-2": {
            "version": 1,
            "slices": [[0, 4], [5, 16]],
            "shape": [4, 4, 128, 128],
            "sig_dims": 2,
            "dtype": "float32",
        },
    }

    sidecar_path = os.path.join(draw_directory, 'structure.json')

    def _scramble_sidecars():
        import socket
        sidecar_data = data[socket.gethostname()]
        print(sidecar_path, os.listdir(os.path.dirname(sidecar_path)))
        with open(sidecar_path, "w") as fh:
            json.dump(sidecar_data, fh)
        print(sidecar_path, os.listdir(os.path.dirname(sidecar_path)))

    print(
        list(dist_ctx.executor.run_each_host(_scramble_sidecars))
    )

    print(draw_directory)

    dirlists = dist_ctx.executor.run_each_host(lambda: os.listdir(draw_directory))
    for v in dirlists.values():
        print(set(v))

    sidecars = dist_ctx.executor.run_each_host(lambda: open(sidecar_path).read())
    print(sidecars)

    # try to re-open the dataset, should fail:
    ds = ClusterDataSet(path=draw_directory, structure=structure)

    with pytest.raises(DataSetException) as e:
        ds = ds.initialize(dist_ctx.executor)

    assert e.match('inconsistent sidecars, please inspect')


@pytest.mark.dist
def test_missing_sidecars_are_created(draw_directory, dist_ctx):
    # tmpdir_factory doesn't output plain strings,
    # and thus draw_directory was some pytest object. we don't have
    # pytest installed on the worker containers, so it failed spectacularly:
    draw_directory = str(draw_directory)
    data = {
        "version": 1,
        "slices": [[0, 8], [16, 256]],
        "shape": [4, 4, 128, 128],
        "sig_dims": 2,
        "dtype": "float32",
    }
    structure = PartitionStructure.from_json(data)
    ds = ClusterDataSet(path=draw_directory, structure=structure)
    ds = ds.initialize(dist_ctx.executor)
    ds.check_valid()

    dirlists = dist_ctx.executor.run_each_host(lambda: os.listdir(draw_directory))

    for v in dirlists.values():
        assert set(v) == {"parts", "structure.json"}
