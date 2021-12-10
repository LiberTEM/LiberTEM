_unpackable_types = (list, tuple, dict)
merge_fns = {list: lambda lis, el, pos: lis.append(el),
             dict: lambda dic, el, pos: dic.update({pos: el})}


class IgnoreClass:
    pass


class StructDescriptor:
    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f'{self.__class__.__name__}({self.cls}, {self.args}, {self.kwargs})'


def flatten_nested(el, unpackable_types=None, ignore_types=None):
    if unpackable_types is None:
        unpackable_types = _unpackable_types
    if ignore_types is None:
        ignore_types = (IgnoreClass,)
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
    if unpackable_types is None:
        unpackable_types = _unpackable_types
    if ignore_types is None:
        ignore_types = (IgnoreClass,)
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
            # Hack tuples to list to avoid immutability problems
            if nest_class == tuple:
                nest_class = list
            nest = nest_class()
        nest = insert_at_pos(el, coords, nest)
    # Convert hacked lists into tuples, from deepest to shallowest
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


def insert_at_pos(el, coords, nest):
    _nest = nest
    for current_coord, next_coord in pairwise(coords):
        current_cls, current_pos = current_coord
        next_cls, next_pos = None, None
        if next_coord is not None:
            next_cls, next_pos = next_coord
        # Hack tuples to lists to avoid immutability problems
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
        if len(indices) > 0:
            set_as_tuple(nest, indices)
        else:
            nest = tuple(nest)
    return nest
