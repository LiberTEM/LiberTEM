import pathlib
import tomli
import json
import os
import numpy as np

from typing import Dict, Any, Optional, Union
from typing_extensions import Literal

from utils import ParserException, resolve_jsonpath
from utils import format_defs, resolve_path_glob

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


class NestedDict(dict):
    """
    Nested dictionary class with knowledge of its parent in the tree
    Implements two features:

    - the ability to resolve keys within the tree using JSON syntax
      relative to the root
            #/path/to/key
      This could be extended to resolve using posix path semantics
    - The ability to search upwards in the tree for a specific key
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


class TreeFactory:
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
            raise ParserException(f"Unrecognized format {format}")

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
    ParserException. This is required as we need to
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
                raise ParserException('Cannot specify global path to non-dict values')
    tree.update(insertions)
    # Go down one level and apply the resolution to any sub-dicts
    insertions = {}
    for key, value in tree.items():
        if isinstance(value, NestedDict):
            insertions[key] = resolve_paths(value)
    tree.update(insertions)
    return tree


def freeze_tree(tree: NestedDict):
    tree_copy = tree.copy()
    tree_copy = resolve_paths(tree_copy)
    if 'root' not in tree_copy:
        tree_copy['root'] = tree.root.get('root', pathlib.Path())
    return tree_copy.to_dict()


from pydantic import BaseModel, Extra, validator, Field


class WithRootModel(BaseModel):
    root: Optional[pathlib.Path] = Field(default=None, repr=False)

    class Config:
        allow_population_by_field_name = True
        extra = Extra.allow


class FileConfig(WithRootModel):
    config_type: Literal['file'] = 'file'
    path: Union[pathlib.Path, str]
    file_format: Optional[str] = None
    load_options: Dict = Field(default_factory=lambda: {})

    @validator('file_format', pre=True)
    def format_clean(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @validator('file_format')
    def format_is_defined(cls, v):
        if v not in format_defs:
            raise ValueError(f'Format {v} unknown')
        return v

    @classmethod
    def construct(cls, value: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not isinstance(value, dict):
            value = {'path': value}
        return value

    def resolve(self) -> pathlib.Path:
        paths = resolve_path_glob(self.path, self.path_root)
        if len(paths) != 1:
            raise ValueError(f'path {self.path} matched {len(paths)} files')
        return paths[0]

    def load(self) -> np.ndarray:
        if self.file_format is None:
            format = self.resolve().suffix.lstrip('.').lower()
        else:
            format = self.file_format
        if format not in format_defs.keys():
            raise ParserException(f'Unrecognized file format {format}')
        return format_defs[format](self.path, **self.load_options)

class MIBDatasetConfig(WithRootModel):
    config_type: Literal['dataset'] = Field(default='dataset', repr=False)
    ds_format: Literal['mib'] = Field(repr=False)
    mib_path: Union[FileConfig, pathlib.Path, str]
    hdr_path: Union[FileConfig, pathlib.Path, str]

    _cast_file = validator(
        'mib_path',
        'hdr_path',
        pre=True,
        allow_reuse=True
    )(FileConfig.construct)


if __name__ == '__main__':
    config_str = R"""
root='/home/mat/Data/ds1'

[dataset_config]
config_type='dataset'
ds_format='mib'
hdr_path = 'test.hdr'
mib_path='#/my_mib_file'

[my_mib_file]
config_type='file'
path='testpath.mib'
"""

    nest = TreeFactory.from_string(config_str)
    file_config = freeze_tree(nest['dataset_config'])
    ds_model = MIBDatasetConfig(**file_config)
