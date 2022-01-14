from libertem.executor.utils import delayed_unpack


def unpack_example():
    nest = [5, 1, (2, 4), {'a': 7, 'b': 10}, 4, (3, 3), 1]
    flat = [5, 1, 2, 4, 7, 10, 4, 3, 3, 1]
    mapping = [[(list, 0)],
               [(list, 1)],
               [(list, 2), (tuple, 0)],
               [(list, 2), (tuple, 1)],
               [(list, 3), (dict, 'a')],
               [(list, 3), (dict, 'b')],
               [(list, 4)],
               [(list, 5), (tuple, 0)],
               [(list, 5), (tuple, 1)],
               [(list, 6)]]
    return nest, flat, mapping


def test_flatten_1():
    nest, flat, _ = unpack_example()
    flattened = delayed_unpack.flatten_nested(nest)
    assert len(flattened) == len(flat)
    assert all(a == b for a, b in zip(flat, flattened))


def test_mapping_1():
    nest, _, mapping = unpack_example()
    built_map = delayed_unpack.build_mapping(nest)
    assert len(built_map) == len(mapping)
    for built, true in zip(built_map, mapping):
        assert all(a == b for a, b in zip(built, true))


def test_rebuild_1():
    nest, flat, mapping = unpack_example()
    rebuilt = delayed_unpack.rebuild_nested(flat, mapping)
    assert len(rebuilt) == len(nest)
    for el_rebuilt, el_original in zip(rebuilt, nest):
        assert el_rebuilt == el_original


def test_empty_elements():
    ic = delayed_unpack.IgnoreClass
    nest = [(2, 4), {}, (4, []), 5]
    flat = [2, 4, ic, 4, ic, 5]
    mapping = [[(list, 0), (tuple, 0)],
               [(list, 0), (tuple, 1)],
               [(list, 1), (dict, ic)],
               [(list, 2), (tuple, 0)],
               [(list, 2), (tuple, 1), (list, ic)],
               [(list, 3)]]
    flattened = delayed_unpack.flatten_nested(nest)
    assert len(flattened) == len(flat)
    assert all(a == b for a, b in zip(flat, flattened))

    built_map = delayed_unpack.build_mapping(nest)
    assert len(built_map) == len(mapping)
    for built, true in zip(built_map, mapping):
        assert all(a == b for a, b in zip(built, true))

    rebuilt = delayed_unpack.rebuild_nested(flat, mapping)
    assert len(rebuilt) == len(nest)
    for el_rebuilt, el_original in zip(rebuilt, nest):
        assert el_rebuilt == el_original


def test_not_unpackable():
    nest = 5
    flat = [5]
    mapping = [None]

    flattened = delayed_unpack.flatten_nested(nest)
    assert len(flattened) == len(flat)
    assert all(a == b for a, b in zip(flat, flattened))

    built_map = delayed_unpack.build_mapping(nest)
    assert built_map == mapping

    rebuilt = delayed_unpack.rebuild_nested(flat, mapping)
    assert rebuilt == nest
