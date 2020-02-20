from libertem.io.utils import get_partition_shape
from libertem.io.dataset.base.utils import FileTree
from libertem.common import Shape

from utils import MockFile


def test_partition_shape_1d():
    ds_shape = Shape((15, 16, 16), sig_dims=2)
    pshape = get_partition_shape(
        dataset_shape=ds_shape,
        target_size_items=256*1024,
        min_num=2
    )
    assert pshape == (7,)


def test_partition_shape_1():
    assert get_partition_shape(Shape((15, 16, 16), sig_dims=2), target_size_items=512) == (
        (2,)
    )


def test_partition_shape_2():
    assert get_partition_shape(Shape((1, 15, 16, 16), sig_dims=2), target_size_items=512) == (
        (1, 2,)
    )


def test_partition_shape_3():
    assert get_partition_shape(Shape((15, 15, 16, 16), sig_dims=2), target_size_items=15*512) == (
        (2, 15,)
    )


def test_partition_shape_4():
    assert get_partition_shape(
        Shape((128, 15, 15, 16, 16), sig_dims=2),
        target_size_items=15*512
    ) == (
        (1, 2, 15,)
    )


def test_partition_shape_5():
    assert get_partition_shape(
        Shape((2, 16, 16), sig_dims=2),
        target_size_items=512,
        min_num=3
    ) == (
        (1,)
    )


def test_partition_shape_small():
    assert get_partition_shape(Shape((15, 16, 16), sig_dims=2), target_size_items=4) == (
        (1,)
    )


def mock_files(num):
    return [
        MockFile(start_idx=x, end_idx=x + 8)
        for x in range(0, num*8, 8)
    ]


def test_make_file_tree_1():
    files = mock_files(1)
    tree = FileTree.make(files)
    print(tree)

    files = mock_files(2)
    tree = FileTree.make(files)
    print(tree)

    files = mock_files(3)
    tree = FileTree.make(files)
    print(tree)

    files = mock_files(4)
    tree = FileTree.make(files)
    print(tree)

    files = mock_files(5)
    tree = FileTree.make(files)
    print(tree)


def test_search():
    files = mock_files(5)
    tree = FileTree.make(files)
    print(tree)

    for i in range(8):
        assert tree.search_start(i)[1].start_idx == 0

    for i in range(16, 24):
        assert tree.search_start(i)[1].start_idx == 16

    for i in range(24, 32):
        assert tree.search_start(i)[1].start_idx == 24

    for i in range(32, 40):
        assert tree.search_start(i)[1].start_idx == 32


def test_ft_single_frame_single_file():
    f1 = MockFile(start_idx=0, end_idx=1)
    files = [f1]

    ft = FileTree.make(files)

    assert ft.search_start(0) == (0, f1)

    print(ft, ft.__dict__)


def test_ft_single_frame_multiple_files():
    f1 = MockFile(start_idx=0, end_idx=1)
    f2 = MockFile(start_idx=1, end_idx=2)
    f3 = MockFile(start_idx=2, end_idx=3)
    f4 = MockFile(start_idx=3, end_idx=4)
    files = [f1, f2, f3, f4]

    ft = FileTree.make(files)

    assert ft.search_start(0) == (0, f1)

    print(ft, ft.__dict__)


def test_ft_multi_frame_multiple_files_even():
    f1 = MockFile(start_idx=0, end_idx=2)
    f2 = MockFile(start_idx=2, end_idx=4)
    f3 = MockFile(start_idx=4, end_idx=6)
    f4 = MockFile(start_idx=6, end_idx=8)
    files = [f1, f2, f3, f4]

    ft = FileTree.make(files)
    print(ft)

    assert ft.search_start(0) == (0, f1)
    assert ft.search_start(2) == (1, f2)
    assert ft.search_start(4) == (2, f3)
    assert ft.search_start(6) == (3, f4)


def test_ft_multi_frame_multiple_files_odd():
    f1 = MockFile(start_idx=0, end_idx=1)
    f2 = MockFile(start_idx=2, end_idx=3)
    f3 = MockFile(start_idx=4, end_idx=5)
    f4 = MockFile(start_idx=6, end_idx=7)
    f5 = MockFile(start_idx=8, end_idx=9)
    files = [f1, f2, f3, f4, f5]

    ft = FileTree.make(files)
    print(ft)

    assert ft.search_start(0) == (0, f1)
    assert ft.search_start(2) == (1, f2)
    assert ft.search_start(4) == (2, f3)
    assert ft.search_start(6) == (3, f4)
    assert ft.search_start(8) == (4, f5)
