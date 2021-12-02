import numpy as np
import dask
import dask.array as da
from dask import delayed


unpackable_types = (list, tuple, dict)


def flatten_nested(el, unpackable_types=None, ignore_types=None):
    flattened = []
    if isinstance(el, unpackable_types) and not isinstance(el, ignore_types):
        iterable = el.values() if isinstance(el, dict) else el
        for _el in iterable:
            flattened.extend(flatten_nested(_el,
                                            unpackable_types=unpackable_types,
                                            ignore_types=ignore_types))
    else:
        flattened.append(el)
    return flattened


def build_mapping(el, pos=None, unpackable_types=None, ignore_types=None):
    flat_mapping = []
    eltype = type(el)
    if isinstance(el, unpackable_types) and not isinstance(el, ignore_types):
        iterable = el.items() if isinstance(el, dict) else enumerate(el)
        for _pos, _el in iterable:
            if pos is None:
                pos = []
            flat_mapping.extend(build_mapping(_el, pos=pos + [(eltype, _pos)],
                                              unpackable_types=unpackable_types,
                                              ignore_types=ignore_types))
    else:
        flat_mapping.append(pos)
    return flat_mapping


def rebuild_nested(flat, flat_mapping):
    nest = None
    for el, coords in zip(flat, flat_mapping):
        if nest is None:
            nest_class = coords[0][0]
            nest = nest_class()
        # set_at_location(nest, coords, el)
        nest = insert_at_pos(el, coords, nest)
    nest = list_to_tuple(nest, flat_mapping)
    return nest


def pairwise(iterable):
    for el in iterable:
        try:
            yield prior_el, el
        except NameError:
            pass
        prior_el = el
    yield prior_el, None


merge_fns = {tuple: lambda lis, el, pos: lis.append(el),
             list: lambda lis, el, pos: lis.append(el),
             dict: lambda dic, el, pos: dic.update({pos: el})}


def insert_at_pos(el, coords, nest):
    _nest = nest
    for current_coord, next_coord in pairwise(coords):
        current_cls, current_pos = current_coord
        next_cls, next_pos = None, None
        if next_coord is not None:
            next_cls, next_pos = next_coord
        if next_cls == tuple:
            next_cls = list
        if next_pos is None:
            merge_fns[current_cls](_nest, el, current_pos)
        else:
            try:
                _nest = _nest[current_pos]
            except KeyError:
                _nest[current_pos] = next_cls()
                _nest = _nest[current_pos]
            except IndexError:
                _nest.append(next_cls())
                _nest = _nest[current_pos]
    return nest


def find_tuples(flat_mapping):
    return [(i, j) for i, coord in enumerate(flat_mapping)
            for j, _coord in enumerate(coord)
            if _coord[0] == tuple]


def set_at_location(nest, coord, value):
    _, loc = coord[0]
    if len(coord) > 1:
        new_cls, _ = coord[1]
        try:
            set_at_location(nest[loc], coord[1:], value)
        except KeyError:
            nest[loc] = new_cls()
            set_at_location(nest, coord, value)
        except IndexError:
            if isinstance(nest, tuple):
                nest = nest + (new_cls(),)
                set_at_location(nest, coord, value)
            elif isinstance(nest, list):
                nest.append(new_cls())
                set_at_location(nest, coord, value)
            else:
                raise RuntimeError('Unexpected unpackable')
    else:
        try:
            nest[loc] = value
        except TypeError:
            nest = nest + (value,)
        except IndexError:
            nest.append(value)


def set_as_tuple(nest, indices):
    if (len(indices) > 1) and isinstance(nest[indices[0]], (dict, list)):
        set_as_tuple(nest[indices[0]], indices[1:])
    else:
        nest[indices[0]] = tuple(nest[indices[0]])


def list_to_tuple(nest, flat_mapping):
    tuple_positions = find_tuples(flat_mapping)
    deepest_first = reversed(sorted(tuple_positions, key=lambda x: x[1]))
    for coord_i, depth_j in deepest_first:
        indices = [c[1] for c in flat_mapping[coord_i][:depth_j]]
        set_as_tuple(nest, indices)
    return nest


class StructDescriptor(tuple):
    pass


def get_res():
    return {'int': 55, 'arr': np.ones((55, 55), dtype=np.float32)}


if __name__ == '__main__':
    res = delayed(get_res)()

    structure = {'int': (StructDescriptor((int,)), StructDescriptor((float,))),
                 'arr': StructDescriptor((np.ndarray, (55, 55), np.float32)),
                 'test': [5, {'a': (55, 44), 'b': [4, 3, 32]}, 2, (66, 47, (28, 33))],
                 'str': "stringhere"}

    flat_structure = flatten_nested(structure,
                                    unpackable_types=unpackable_types,
                                    ignore_types=(StructDescriptor,))
    flat_mapping = build_mapping(structure,
                                 unpackable_types=unpackable_types,
                                 ignore_types=(StructDescriptor,))

    renested = rebuild_nested(flat_structure, flat_mapping)

    # unpacked = delayed(res_unpack, nout=len(structure))(res, structure)
    # unpacked_delayeds = {key: _delayed for key, _delayed in zip(structure.keys(), unpacked)}
