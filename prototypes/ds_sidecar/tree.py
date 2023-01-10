import pathlib
import tomli
import json
import os

from typing import Dict, Any, Optional, Union, Callable, Tuple
from typing_extensions import Literal


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
        """
        Get the parent NestedDict of self or None if self is the root
        """
        try:
            return self._parent
        except AttributeError:
            return None

    @property
    def root(self) -> 'NestedDict':
        """
        Get the root of the tree in which self sits
        """
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
        return get_from_jsonpath(self.root, key)

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
        """
        Return a bare dict copy of self and all children
        """
        new = {}
        for key, value in self.items():
            if isinstance(value, self.__class__):
                new[key] = value.to_dict()
            else:
                new[key] = value
        return new

    def freeze(self) -> Dict[str, Any]:
        """
        Return a bare dict copy of self and all children

        In doing so resolve any relative paths (#/path/to/key)
        and propagate the 'path root' key from the top of the tree
        """
        return freeze_tree(self)


class TreeFactory:

    @classmethod
    def from_file(
        cls,
        path: os.PathLike,
        root: Optional[os.PathLike] = None,
    ) -> NestedDict:
        """
        Load the configuration data from path and return it as a tree

        If the key 'root' is not present at the top of the config,
        either the supplied root argument will be inserted or the path
        to the config file itself if root is None.

        Does not parse the config at this stage
        """
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
    def from_string(
        cls,
        string: str,
        format: Literal['toml', 'json'] = 'toml',
        root: Optional[os.PathLike] = None
    ) -> NestedDict:
        """
        Load the configuration data from a string using a particular format
        and return it as a tree

        If the key 'root' is not present at the top of the config,
        either the supplied root argument will be inserted or the current
        working directory if root is None.

        Does not parse the config at this stage
        """
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
    def to_tree(cls, struct: Dict[str, Any]) -> NestedDict:
        """
        Convert the dict struct to NestedDict tree
        """
        if not isinstance(struct, dict):
            raise TypeError('Can only convert dict-like to tree')
        if 'root' not in struct:
            raise ValueError('Need a "root" key at top level to define relative paths')
        return build_tree(struct)


def build_tree(struct: Dict[str, Any], parent=None):
    """
    Recurse through struct converting any dict instances
    to NestedDict while setting the parent attribute
    """
    struct = NestedDict(**struct)
    struct._set_parent(parent)
    for key, value in struct.items():
        if isinstance(value, dict):
            struct[key] = build_tree(value, parent=struct)
    return struct


def get_from_jsonpath(struct: Dict[str, Any], jsonpath: str):
    """
    Resolve a path in JSON notation #/path/to/key
    down from struct and return the value found at that path
    """
    if not isinstance(jsonpath, str):
        raise TypeError(f'Key to resolve must be string, got {jsonpath}')
    components = jsonpath.strip().strip('#/').split('/')
    components = list(c for c in components if len(c) > 0)
    return get_in_struct(struct, components)


def get_in_struct(struct: Dict[str, Any], components: Tuple[str]):
    """
    Generic function to resolve a path in nested struct using
    keys sequentially taken from components

    An empty components tuple will return struct
    """
    view = struct
    for c in components:
        if not isinstance(view, dict):
            raise KeyError(f'Cannot access key {c} in {type(view)} from path {components}')
        elif c not in view:
            raise KeyError(f'Cannot resolve key {components} in struct, missing {c}')
        view = view.get(c)
    return view


def resolve_references(tree: NestedDict):
    """
    Resolve JSON-path references by converting from
    string #/ to the value found at that path

    If the referenced value is not a NestedDict raises
    TypeError. This is required as we need to
    retain the original parent to get the correct file
    root semantics. There is no easy way to give any
    arbitrary value a parent attribute.

    #FIXME Could get into an infinite loop if a set of paths form a cycle

    Modifies tree inplace so should run on a copy
    """
    # First resolve relative keys at this level
    # By replacing at this level we keep any NestedDicts
    # pointing at their original parents
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
            insertions[key] = resolve_references(value)
    tree.update(insertions)
    return tree


def propagate_key(tree: NestedDict, key: str) -> NestedDict:
    """
    Copy a key and its value down in tree to all child NestedDict

    If a child defines its own key, propagate this to its children

    Tree must contain key
    """
    if key not in tree:
        raise KeyError(f'Need key: {key} in tree to propagate down')
    _propagate_key(tree, tree[key])
    return tree


def _propagate_key(tree: NestedDict, key: str, parent_value):
    """
    Recurse into tree setting key to parent_value if not already present
    If present, set the new value on children
    """
    tree.setdefault(key, parent_value)
    for value in tree.values():
        if isinstance(value, NestedDict):
            _propagate_key(value, tree[key])


def freeze_tree(tree: NestedDict) -> Dict[str, Any]:
    """
    Make a copy of tree as a plain dictionary with any JSON-path
    references resolved and the 'root' key propagated to all children
    """
    tree_copy = tree.copy()
    tree_copy = propagate_key(tree_copy.root, 'root')
    tree_copy = resolve_references(tree_copy)
    return tree_copy.to_dict()


def find_in_tree(tree: NestedDict, matches: Union[Dict[str, Any], Callable[[NestedDict], bool]]):
    """
    Yield all sub-trees from tree satisfying matches

    matches can either be a dictionary of key/value pairs
    which must be present in the sub-tree to validate, or a callable
    taking the sub-tree and returning bool to yield the sub-tree
    """
    if does_match(tree, matches):
        yield tree
    for v in tree.values():
        if isinstance(v, NestedDict):
            yield from find_in_tree(v, matches)


def does_match(tree: NestedDict, matches: Union[Dict[str, Any],
                                                Callable[[NestedDict], bool]]) -> bool:
    """
    Checks tree against the matches argument

    If matches is a dict of key/value pairs to check,
    these must all support the equality operator
    """
    if isinstance(matches, dict):
        if all(k in tree and tree[k] == v for k, v in matches.items()):
            return True
    elif callable(matches):
        if matches(tree):
            return True
    else:
        raise TypeError(f'Cannot use {type(matches)} type object to search tree')
    return False
