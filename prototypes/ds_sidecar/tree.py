import pathlib
import tomli
import json
import os

from typing import Dict, Any, Optional, Union, Callable


class NestedDict(dict):
    """
    Nested dictionary class with knowledge of its parent in the tree

    Implements the ability to resolve keys within the tree using
    JSON syntax relative to the root
            #/path/to/key
    """
    def _set_parent(self, parent: Dict[str, Any]):
        self._parent = parent

    @property
    def parent(self):
        try:
            return self._parent
        except AttributeError:
            return None

    @property
    def root(self) -> 'NestedDict':
        parent = self.parent
        if parent is not None:
            return parent.root
        return self

    def resolve_key(self, key: str):
        """
        Get key from tree in JSON path notation
           i.e. #/key1/key2
        starting from root
        If not available then raise
        """
        if not isinstance(key, str):
            raise TypeError(f'Invalid key {key}')
        if not key.startswith('#/'):
            raise KeyError(f'Can only resolve keys in JSON-path syntax (#/), got {key}')
        return resolve_jsonpath(self.root, key)

    def copy(self):
        """
        Return a copy of this NestedDict instance where
        all the .parent references also point to copies in a new tree
        """
        root = self.root
        new_root = root._copy_down()
        return new_root.resolve_key(self.where)

    def _copy_down(self):
        new = self.__class__(**self)
        copy = {}
        for key, value in self.items():
            try:
                value = value._copy_down()
                value._set_parent(new)
            except AttributeError:
                pass
            copy[key] = value
        new.update(copy)
        return new

    @property
    def where(self) -> str:
        """
        Get the JSON path #/ from root for this struct
        """
        parent = self.parent
        if parent is not None:
            me, = tuple(k for k, v in parent.items() if v is self)
            me = f'{parent.where}/{me}'
        else:
            me = '#/'
        return me

    def to_dict(self) -> Dict[str, Any]:
        new = {}
        for key, value in self.items():
            if isinstance(value, self.__class__):
                new[key] = value.to_dict()
            else:
                new[key] = value
        return new

    def freeze(self) -> Dict[str, Any]:
        return freeze_tree(self)


class TreeFactory:

    @classmethod
    def from_file(cls, path, root: Optional[os.PathLike] = None):
        path = pathlib.Path(path)

        if not path.is_file():
            raise FileNotFoundError(f"Cannot find spec file {path}")

        if path.suffix == '.toml':
            with path.open('rb') as fp:
                struct = tomli.load(fp)
        elif path.suffix == '.json':
            with path.open('r') as fp:
                struct = json.load(fp)
        # elif path.suffix == '.yaml':
        #     ...
        else:
            raise ValueError(f"Unrecognized format {path.suffix}")

        # if no top-level root set the parent directory of config file
        if root is None:
            struct.setdefault('root', path.parent)
        else:
            struct['root'] = root

        return cls.to_tree(struct)

    @classmethod
    def from_string(cls, string, format='toml', root: Optional[os.PathLike] = None):
        if format == 'toml':
            struct = tomli.loads(string)
        elif format == 'json':
            struct = json.loads(string)
        else:
            raise ValueError(f"Unrecognized format {format}")

        if root is None:
            struct.setdefault('root', pathlib.Path())
        else:
            struct['root'] = root
        return cls.to_tree(struct)

    @classmethod
    def to_tree(cls, struct: Dict[str, Any]):
        if 'root' not in struct:
            raise ValueError('Need a "root" key at top level to define relative paths')
        return build_tree(struct)


def build_tree(struct: Dict[str, Any], parent=None):
    if not isinstance(struct, dict):
        return struct
    struct = NestedDict(**struct)
    struct._set_parent(parent)
    for key, value in struct.items():
        if isinstance(value, dict):
            struct[key] = build_tree(value, parent=struct)
    return struct


def resolve_paths(tree: NestedDict):
    """
    Resolve JSON-type paths by converting from
    string #/ to the value found at that path

    If the referenced value is not a NestedDict raises
    TypeError. This is required as we need to
    retain the original parent to get the correct file
    root semantics. There is no easy way to give any
    arbitrary value a parent attribute.

    #FIXME Could get into an infinite loop if a set of paths form a cycle

    Modifies tree inplace
    """
    # First resolve relative keys at this level
    # By replacing at this level we keep any NestedDicts
    # pointing at their original parents / root
    insertions = {}
    for key, value in tree.items():
        if isinstance(value, str) and value.startswith('#/'):
            insertions[key] = tree.resolve_key(value)
            if not isinstance(insertions[key], NestedDict):
                raise TypeError('Cannot specify global path to non-dict values')
    tree.update(insertions)
    # Go down one level and apply the resolution to any sub-dicts
    insertions = {}
    for key, value in tree.items():
        if isinstance(value, NestedDict):
            insertions[key] = resolve_paths(value)
    tree.update(insertions)
    return tree


def propagate_path_root(tree: NestedDict) -> NestedDict:
    tree_root = tree.root
    if 'root' not in tree_root:
        tree_root['root'] = pathlib.Path()
    _propagate_path_root(tree_root, tree_root['root'])
    return tree


def _propagate_path_root(tree: NestedDict, parent_root: pathlib.Path):
    tree.setdefault('root', parent_root)
    for value in tree.values():
        if isinstance(value, NestedDict):
            _propagate_path_root(value, tree['root'])


def freeze_tree(tree: NestedDict):
    tree_copy = tree.copy()
    tree_copy = propagate_path_root(tree_copy)
    tree_copy = resolve_paths(tree_copy)
    if 'root' not in tree_copy:
        tree_copy['root'] = tree.root.get('root', pathlib.Path())
    return tree_copy.to_dict()


def resolve_jsonpath(struct: Dict, jsonpath: str):
    return _resolve_generic(struct, jsonpath, '#/', '/')


def _resolve_generic(struct: Dict, path: str, strip: str, split: str):
    if not isinstance(path, str):
        raise TypeError(f'Cannot resolve key {path}')
    components = path.strip().strip(strip).split(split)
    components = list(c for c in components if len(c) > 0)
    view = struct
    for c in components:
        if not isinstance(view, dict) or c not in view:
            raise KeyError(f'Cannot resolve key {path}')
        view = view.get(c)
    return view


def find_in_tree(tree: NestedDict, matches: Union[Dict[str, Any], Callable[[Any], bool]]):
    root = tree.root
    yield from _find_in_tree(root, matches)


def _find_in_tree(tree: NestedDict, matches: Union[Dict[str, Any], Callable[[Any], bool]]):
    if does_match(tree, matches):
        yield tree
    for v in tree.values():
        if isinstance(v, NestedDict):
            yield from _find_in_tree(v, matches)


def does_match(tree: NestedDict, matches: Union[Dict[str, Any],
                                                Callable[[NestedDict], bool]]) -> bool:
    if isinstance(matches, dict):
        if all(k in tree and tree[k] == v for k, v in matches.items()):
            return True
    elif callable(matches):
        if matches(tree):
            return True
    else:
        raise TypeError(f'Cannot use {type(matches)} type object to search tree')
    return False
