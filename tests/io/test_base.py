from collections import namedtuple


from libertem.io.dataset.base import FileTree


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
    l = print_tree(tree.l, indent + our_len)
    r = print_tree(tree.r, indent + our_len)
    s = " " * (indent + our_len)
    ret = f"\n{s}{l}\n{s}{r})"
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
