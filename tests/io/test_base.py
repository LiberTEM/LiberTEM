from collections import namedtuple

import numpy as np
import pytest

from libertem.io.dataset import get_extensions
from libertem.io.dataset.base import (
    FileTree, Partition3D, _roi_to_nd_indices
)
from libertem.common import Shape, Slice
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random

FakeFile = namedtuple('FakeFile', ['start_idx', 'end_idx'])


def mock_files(num):
    return [
        FakeFile(start_idx=x, end_idx=x + 8)
        for x in range(0, num*8, 8)
    ]


def print_tree(tree, indent=0):
    if tree is None:
        return "-"
    ret_us = f"([{tree.low}, {tree.high}] "
    our_len = len(ret_us)
    leh = print_tree(tree.left, indent + our_len)
    r = print_tree(tree.right, indent + our_len)
    s = " " * (indent + our_len)
    ret = f"\n{s}{leh}\n{s}{r})"
    return ret_us + ret


def test_make_file_tree_1():
    files = mock_files(1)
    tree = FileTree.make(files)
    print(print_tree(tree))

    files = mock_files(2)
    tree = FileTree.make(files)
    print(print_tree(tree))

    files = mock_files(3)
    tree = FileTree.make(files)
    print(print_tree(tree))

    files = mock_files(4)
    tree = FileTree.make(files)
    print(print_tree(tree))

    files = mock_files(5)
    tree = FileTree.make(files)
    print(print_tree(tree))


def test_search():
    files = mock_files(5)
    tree = FileTree.make(files)
    print(print_tree(tree))

    for i in range(8):
        assert tree.search_start(i)[1].start_idx == 0

    for i in range(16, 24):
        assert tree.search_start(i)[1].start_idx == 16

    for i in range(24, 32):
        assert tree.search_start(i)[1].start_idx == 24

    for i in range(32, 40):
        assert tree.search_start(i)[1].start_idx == 32


def test_sweep_stackheight():
    data = _mk_random(size=(16, 16, 16, 16))
    for stackheight in range(1, 256):
        print("testing with stackheight", stackheight)
        dataset = MemoryDataSet(
            data=data.astype("<u2"),
            tileshape=(stackheight, 16, 16),
            num_partitions=2,
        )
        for p in dataset.get_partitions():
            for tile in p.get_tiles():
                pass


def test_num_part_larger_than_num_frames():
    shape = Shape((1, 1, 256, 256), sig_dims=2)
    slice_iter = Partition3D.make_slices(shape=shape, num_partitions=2)
    next(slice_iter)
    with pytest.raises(StopIteration):
        next(slice_iter)


def test_roi_to_nd_indices():
    roi = np.full((5, 5), False)
    roi[1, 2] = True
    roi[2, 1:4] = True
    roi[3, 2] = True

    part_slice = Slice(
        origin=(2, 0, 0, 0),
        shape=Shape((2, 5, 16, 16), sig_dims=2)
    )

    assert list(_roi_to_nd_indices(roi, part_slice)) == [
        (2, 1), (2, 2), (2, 3),
                (3, 2)
    ]

    part_slice = Slice(
        origin=(0, 0, 0, 0),
        shape=Shape((5, 5, 16, 16), sig_dims=2)
    )

    assert list(_roi_to_nd_indices(roi, part_slice)) == [
                (1, 2),
        (2, 1), (2, 2), (2, 3),
                (3, 2)
    ]


def test_get_extensions():
    exts = get_extensions()
    assert len(exts) >= 15
    assert "mib" in exts
    assert "gtg" in exts
    # etc...
