from typing import (
    Any, Callable, Optional,
)
from collections.abc import Iterable, Generator


def default_unpackable() -> dict[type, Callable[[Iterable], Iterable[tuple[Any, Any]]]]:
    """
    Defaults for types which can be unpacked by the
    functions in this file, providing a mapping from
    type to a fn(instance) giving an iterable yielding
    (index, element) within the unpackable.
    """
    _unpackable_types = {list: lambda x: enumerate(x),
                        tuple: lambda x: enumerate(x),
                        dict: lambda x: x.items()}
    return _unpackable_types


def default_merge_fns() -> dict[type, Callable[[Iterable, Any, Any], None]]:
    """
    Default merge functions for rebuilding structures

    Returns a mapping from {type: callable} with callable signature:
        fn(structure, el, position)
    which inserts el into the structure at position
    """
    merge_fns = {list: lambda lis, el, pos: lis.append(el),
                dict: lambda dic, el, pos: dic.update({pos: el})}
    return merge_fns


class IgnoreClass:
    """
    A Null type used as a signal when None or False have meaning
    """
    pass


class StructDescriptor:
    def __init__(self, cls: type, *args: tuple, **kwargs: dict):
        """
        Container to hold a reference to a type and both the
        args and kwargs necessary to initialise it

        Could be replaced with a simple dict, namedtuple etc but being
        able to *args, **kwargs for variable numbers of parameters
        is quite useful
        """
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def __repr__(self):
        return f'{self.__class__.__name__}({self.cls}, {self.args}, {self.kwargs})'


def flatten_nested(el: Any,
                   unpackable_types: dict[type, Callable[[Iterable],
                                                         Iterable[tuple[Any, Any]]]] = None,
                   ignore_types: Optional[tuple[type, ...]] = None) -> list[Any]:
    """
    Recursively unpack the structure el while the type of el is in
    the mapping unpackable_types, which maps between the types that
    can be unpacked and a function fn(el) returning an iterable
    that gives tuples of (index, subelement),
    e.g. enumerate(['a', 'b']) => (0, 'a'), (1, 'b')

    ignore_types are types which this function should specifically ignore
    and not try to unpack, even if they are present in unpackable_types

    Returns a flat list containing all the elements of the
    (possibly nested) structure el
    """
    eltype = type(el)
    if unpackable_types is None:
        unpackable_types = default_unpackable()
    if ignore_types is None:
        ignore_types = (IgnoreClass,)
    flattened = []
    if eltype in unpackable_types.keys() and not isinstance(el, ignore_types):
        # If el is unpackable but empty
        if not el:
            flattened.append(IgnoreClass)
            return flattened
        iterable = unpackable_types[eltype](el)
        for _, _el in iterable:
            flattened.extend(flatten_nested(_el,
                                            unpackable_types=unpackable_types,
                                            ignore_types=ignore_types))
    else:
        flattened.append(el)
    return flattened


def build_mapping(el: Any,
                  unpackable_types: dict[type, Callable[[Iterable],
                                                        Iterable[tuple[Any, Any]]]] = None,
                  ignore_types: Optional[tuple[type, ...]] = None,
                  _pos: list[tuple[type, Any]] = None) -> list[list[tuple[type, Any]]]:
    """
    Recursively unpack the structure el and build a flat descriptor of its
    structure, such that it can be re-built

    The elements of the return list are essentially each a list of 'coordinates'
    that map from a position in the flattened version of el, to the position
    in the (possibly nested) original structure el

    Same arguments as flatten_nested except pos, which is only used by the
    function as a way to pass the current position down to the next
    level of the function calls
    """
    flat_mapping = []
    eltype = type(el)
    if unpackable_types is None:
        unpackable_types = default_unpackable()
    if ignore_types is None:
        ignore_types = (IgnoreClass,)
    if eltype in unpackable_types.keys() and not isinstance(el, ignore_types):
        # If el is unpackable but empty
        if not el:
            flat_mapping.append(_pos + [(eltype, IgnoreClass)])
            return flat_mapping
        iterable = unpackable_types[eltype](el)
        for __pos, _el in iterable:
            if _pos is None:
                _pos = []
            flat_mapping.extend(build_mapping(_el, _pos=_pos + [(eltype, __pos)],
                                              unpackable_types=unpackable_types,
                                              ignore_types=ignore_types))
    else:
        flat_mapping.append(_pos)
    return flat_mapping


def rebuild_nested(flat: list[Any],
                   flat_mapping: list[list[tuple[type, Any]]],
                   merge_functions=None):
    """
    Using the flattened version of a structure built by flatten_nested
    and the coordinates created by build_mapping, reconstruct the original
    nested structure

    merge_functions is a mapping from type: fn() with signature:
        fn(_nest, el, position)
    which inserts el into the structure _nest at position

    By default this function only knows how to rebuild a nest
    consisting of [list, dict, tuple], and in the tuple case actually
    reconstructs as list before casting to tuple at the end (to avoid
    immutability of tuples). In principle, by supplying extra merge_functions
    this function should be able to reconstruct other mutable iterables.

    This function works left-to-right in the list flat.
    Could perhaps be done better by building from deepest
    to shallowest across the set of elements in flat.
    """
    if merge_functions is None:
        merge_functions = default_merge_fns()
    nest = None
    for el, coords in zip(flat, flat_mapping):
        # Build the outer iterable of the structure
        if nest is None:
            # Case of a non-unpackable initial element
            if coords is None:
                return el
            nest_class = coords[0][0]
            # Hack tuples to list to avoid immutability problems
            if nest_class == tuple:
                nest_class = list
            nest = nest_class()
        nest = insert_at_pos(el, coords, nest, merge_functions)
    # Convert hacked lists into tuples, from deepest to shallowest
    nest = list_to_tuple(nest, flat_mapping)
    return nest


def pairwise(iterable: Iterable[Any]) -> Generator[tuple[Any, Any], None, None]:
    """
    Yield elements of iterable as tuples of overlapping pairs
    finally yielding (last_element, None)
    """
    for el in iterable:
        try:
            yield prior_el, el
        except NameError:
            pass
        prior_el = el
    yield prior_el, None


def insert_at_pos(el: Any,
                  coords: list[tuple[type, Any]],
                  nest: Iterable,
                  merge_fns: dict[type, Callable[[Iterable, Any, Any], None]]):
    """
    For the partially completed nested structure nest, insert the
    element el at the position given by coords

    If the position of el does not exist yet, build the structure
    from the top down until el can be inserted

    merge el into existing structures using a function from
    the mapping merge_fns[type(el)](_nest, el, position)

    tuples are treated as lists to allow appending, and are later
    converted to tuples once the nest is completed
    """
    _nest = nest
    for current_coord, next_coord in pairwise(coords):
        current_cls, current_pos = current_coord
        next_cls, next_pos = None, None
        if next_coord is not None:
            next_cls, next_pos = next_coord
        # Handle empty structures, these are marked with a special
        # coordinate (cls, IgnoreClass) and by definition are the
        # end of insertions for this set of coords
        if next_pos is IgnoreClass:
            merge_fns[type(_nest)](_nest, next_cls(), current_pos)
            return nest
        # Hack tuples to lists to avoid immutability problems
        if next_cls == tuple:
            next_cls = list
        if next_pos is None:
            merge_fns[type(_nest)](_nest, el, current_pos)
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


def find_tuples(flat_mapping: list[list[tuple[type, Any]]]) -> list[tuple[int, int]]:
    """
    Get the indexes in flat_mapping and depth in the coordinate
    where the coordinate specify the structure is of class tuple
    """
    return [(i, j) for i, coord in enumerate(flat_mapping)
            for j, _coord in enumerate(coord)
            if _coord[0] == tuple]


def set_as_tuple(nest, indices: list[Any]):
    """
    For a given sequence of indices to index into the completed
    nest, convert the structure at the final index in the sequence
    to a tuple type (if it is not already a tuple)
    """
    if (len(indices) > 1) and isinstance(nest[indices[0]], (dict, list)):
        set_as_tuple(nest[indices[0]], indices[1:])
    else:
        nest[indices[0]] = tuple(nest[indices[0]])


def list_to_tuple(nest, flat_mapping: list[list[tuple[type, Any]]]):
    """
    Convert any elements which are marked as tuples in flat_mapping
    but were constructed in nest as lists, back into tuples
    """
    tuple_positions = find_tuples(flat_mapping)
    deepest_first = reversed(sorted(tuple_positions, key=lambda x: x[1]))
    for coord_i, depth_j in deepest_first:
        indices = [c[1] for c in flat_mapping[coord_i][:depth_j]]
        if len(indices) > 0:
            set_as_tuple(nest, indices)
        else:
            nest = tuple(nest)
    return nest
