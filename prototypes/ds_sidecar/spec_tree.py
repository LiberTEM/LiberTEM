import pathlib
import tomli
import json
import os

from typing import Dict, Any, Optional

from utils import ParserException
from config_base import SpecBase, NestedDict

import specs
import wrapped_types


spec_types = {
    t.spec_type: t for t in (
        specs.FileSpec,
        specs.FileSetSpec,
        specs.ArraySpec,
        specs.ROISpec,
        specs.CorrectionSetSpec,
        specs.DataSetSpec,
    )
}
extra_types = {
    t.spec_type: t for t in (
        wrapped_types.DType,
    )
}
all_types = {
    **spec_types,
    **extra_types,
}


class SpecTree(SpecBase):
    spec_type = 'tree'

    @classmethod
    def from_file(cls, path, root: Optional[os.PathLike] = None):
        path = pathlib.Path(path)

        if not path.is_file():
            raise ParserException(f"Cannot find spec file {path}")

        if path.suffix == '.toml':
            with path.open('rb') as fp:
                struct = tomli.load(fp)
        elif path.suffix == '.json':
            with path.open('r') as fp:
                struct = json.load(fp)
        # elif path.suffix == '.yaml':
        #     ...
        else:
            raise ParserException(f"Unrecognized format {path.suffix}")

        # if no top-level root set the parent directory of config file
        if root is None:
            root = struct.get('root', path.parent)

        return cls.to_tree(struct, root=root)

    @classmethod
    def from_string(cls, string, format='toml', root: Optional[os.PathLike] = None):
        if format == 'toml':
            struct = tomli.loads(string)
        elif format == 'json':
            struct = json.loads(string)
        else:
            raise ParserException(f"Unrecognized format {format}")
        return cls.to_tree(struct, root=root)

    @classmethod
    def to_tree(cls, struct: Dict[str, Any], root: Optional[os.PathLike] = None):
        # if no top-level root set the CWD
        if root is None:
            root = struct.get('root', pathlib.Path())
        return build_tree(struct, root=root)


def build_tree(struct: Dict[str, Any], root=None, parent=None):
    if not isinstance(struct, dict):
        return struct
    struct = NestedDict(**struct)
    struct._set_parent(parent)
    # struct._set_path_root(root)
    for key, value in struct.items():
        if isinstance(value, dict):
            struct[key] = build_tree(value, parent=struct)
    return struct


# def resolve_from_schema(tree: NestedDict, schema: Dict[str, Any]):
#     """
#     Use a schema to build a complete config from (some) elements of tree

#     Will resolve references/json-paths within the tree
#     Will set the path ._root on each config object in the tree
#     unless the object has a 'root' key which takes precedence
#     If an object has a 'type' key which differs from that required
#     by the schema, raise ValidationError.
#     Will ignore elements of tree which are not specified in the schema
#     but children will be retained and resolved

#     Will cast objects to the correct Types (which may cause a
#     ValidationError to be raised as they may self-validate).
#     This may involve constructing dict-like spec Types
#     from previously single-value keys.
#     The returned config will have no .parent attribute / tree behaviour

#     Does not actually validate the schema, rather constructs the full config
#     which can then be validated in the next step
#     """
#     schema_type = schema.get('type')
#     if schema_type is None:
#         raise ParserException('Need schema with top-level type to resolve')
#     if schema_type not in spec_types:
#         raise ParserException(f'Unrecognized schema type {schema_type}')
#     tree_type = tree.get('type')
#     if tree_type is not None and schema_type != tree_type:
#         raise ParserException('Config defines type not matching schema')
#     spec = spec_types[schema_type](**tree)
#     # if root not in 
#     spec._set_root(tree.path_root)


# def as_type(self, spec_type):
#     ...


# def child_as_type(self, key, spec_type):
#     ...





def resolve_paths(tree: NestedDict):
    """
    Resolve JSON-type paths by converting from
    string #/ to the value found at that path

    If the referenced value is not a NestedDict raises
    ParserException. This is required as we need to
    retain the original parent to get the correct file
    root semantics. There is no easy way to give any
    arbitrary value a parent attribute.

    #FIXME Could get into an infinite loop if a set of paths form a cycle

    Modifies the tree inplace
    """
    # First resolve relative keys at this level
    # By replacing at this level we keep any NestedDicts
    # pointing at their original parents / root
    insertions = {}
    for key, value in tree.items():
        if isinstance(value, str) and value.startswith('#/'):
            insertions[key] = tree.resolve_key(value)
            if not isinstance(insertions[key], NestedDict):
                raise ParserException('Cannot specify global path to non-dict values')
    tree.update(insertions)
    # Go down one level and apply the resolution to any sub-dicts
    insertions = {}
    for key, value in tree.items():
        if isinstance(value, NestedDict):
            insertions[key] = resolve_paths(value)
    tree.update(insertions)
    return tree


def get_from_schema(schema, key):
    _subschema = schema
    for pos in key:
        try:
            _subschema = _subschema['properties'][pos]
        except KeyError:
            return None
    return _subschema


def depth_traversal(tree, pos=None):
    if pos is None:
        pos = tuple()
    for key, value in tree.items():
        if isinstance(value, dict):
            yield from depth_traversal(value, pos=pos + (key,))
        else:
            yield pos, value


def insert_in_tree(tree, key, value):
    _tree = tree
    for pos in key[:-1]:
        _tree = tree[pos]
    _tree[key[-1]] = value


def interpret_with_schema(tree: NestedDict, schema: Dict[str, Any]):
    for pos, value in depth_traversal(tree):
        subschema = get_from_schema(schema, pos)
        if subschema is None:
            continue
        schema_type = subschema.get('type')
        if schema_type is None:
            raise ParserException('Need schema with type to interpret config')
        if isinstance(value, NestedDict):
            value_type = value.get('type')
            if value_type is not None and schema_type != value_type:
                raise ParserException('Config defines type not matching schema')
        if schema_type in all_types:
            # Convert value to the intended type
            spec_cls = all_types[schema_type]
            # spec_cls.construct should do validation if needed
            # and set the path root as a hard attribute on spec
            cast_value = spec_cls.construct(value)
            insert_in_tree(tree, pos, cast_value)


# def interpret_with_schema(tree: NestedDict, schema: Dict[str, Any]):
#     schema_type = schema.get('type')
#     if schema_type is None:
#         raise ParserException('Need schema with type key to interpret config')
#     if isinstance(tree, NestedDict):
#         tree_type = tree.get('type')
#         if tree_type is not None and schema_type != tree_type:
#             raise ParserException('Config defines type not matching schema')

#     # properties must be of a simple form
#     # cannot support conditional schemas
#     properties = schema.get('properties')
#     insertions = {}
#     if properties is not None:
#         for prop_name, prop_schema in properties.items():
#             if prop_name in tree:
#                 insertions[prop_name] = interpret_with_schema(tree[prop_name], prop_schema)
#     tree.update(insertions)

#     # Cast the outer tree to its type if we have a definition for it
#     # else leave the tree / value untouched
#     if schema_type in all_types:
#         # Convert tree to the intended type
#         spec_cls = all_types[schema_type]
#         # spec_cls.construct should do validation if needed
#         # and set the path root as a hard attribute on spec
#         tree = spec_cls.construct(tree)

#     return tree


if __name__ == '__main__':
    ...







    


