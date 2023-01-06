import pathlib
import tomli
import json
import os
import math
import numpy as np
import natsort
import functools
import operator

from typing import Dict, Any, Optional, Union, List
from typing_extensions import Literal

from utils import resolve_jsonpath, resolve_path_glob
from utils import format_defs, sort_methods, format_T


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


class TreeFactory:
    spec_type = 'tree'

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


from pydantic import BaseModel, Extra, validator, root_validator
from pydantic import conlist, PositiveInt, Field

class WithExtraModel(BaseModel):
    class Config:
        allow_population_by_field_name = True
        extra = Extra.allow


class WithRootModel(WithExtraModel):
    root: Optional[pathlib.Path] = Field(default=pathlib.Path(), repr=False)


class FileConfig(WithRootModel):
    config_type: Literal['file'] = Field(default='file', repr=False)
    path: pathlib.Path
    format: Optional[format_T] = None
    load_options: Dict = Field(default_factory=lambda: {})

    @validator('format', pre=True)
    def format_clean(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

    @validator('format')
    def format_is_defined(cls, v):
        if v not in format_defs:
            raise ValueError(f'Format {v} unknown')
        return v

    @classmethod
    def from_value(
        cls,
        path_or_config: Union[str, Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        if not isinstance(path_or_config, dict):
            path_or_config = {
                'path': path_or_config,
                'root': kwargs.get('values', {}).get('root', pathlib.Path())
            }
        return path_or_config

    def resolve(self) -> pathlib.Path:
        paths = resolve_path_glob(self.path, self.root)
        if len(paths) != 1:
            raise ValueError(f'Single path {self.path} matched {len(paths)} files')
        return paths[0]

    def load(self, path: Optional[pathlib.Path] = None) -> np.ndarray:
        if path is None:
            path = self.resolve()
        format = self.format
        if format is None:
            format = path.suffix.lstrip('.').lower()
        if format not in format_defs.keys():
            raise ValueError(f'Unrecognized file format {format}')
        return format_defs[format](path, **self.load_options)


class FileSetConfig(WithRootModel):
    config_type: Literal['fileset'] = Field(default='fileset', repr=False)
    files: Union[List[pathlib.Path], pathlib.Path]
    sort: Optional[Literal['natsorted', 'os_sorted', 'humansorted', 'none']] = 'natsorted'
    sort_options: Optional[List[natsort.ns]] = None

    @validator('sort_options', pre=True, each_item=True)
    def convert_sort_keys(cls, v):
        try:
            v = natsort.ns[v]
        except KeyError:
            raise ValueError(f'Unrecognized sort option {v}')
        return v

    @validator('sort_options')
    def sort_options_no_sort(cls, value, values):
        if value is not None and values.get('sort') is None:
            raise ValueError('Cannot define sort options without sort method')
        return value

    @classmethod
    def from_value(
        cls,
        files_or_config: Union[str, Dict[str, Any]],
        **kwargs
    ) -> Dict[str, Any]:
        if not isinstance(files_or_config, dict):
            files_or_config = {
                'files': files_or_config,
                'root': kwargs.get('values', {}).get('root', pathlib.Path())
            }
        return files_or_config

    def resolve(self):
        if isinstance(self.files, (str, pathlib.Path)):
            filelist = resolve_path_glob(self.files, self.root)
        elif isinstance(self.files, (list, tuple)):
            # List of (potentially mixed) absolute, relative, or glob specifiers
            filelist = [f for path in self.files for f in resolve_path_glob(path, self.path_root)]
        else:
            raise ValueError(f'Unrecognized files specifier {self.files}')

        if not filelist:
            raise RuntimeError(f'Found no files with specifier {self.files}.')

        # It's possible that multiple globs together may match a file more than once
        # Could add some form of uniqueness check for resolved paths ?
        if self.sort:
            filelist = self._sort(filelist)

        return filelist

    def _sort(self, filelist):
        sort_fn = sort_methods.get(self.sort)
        if sort_fn is None:
            return filelist
        alg_option = natsort.ns.DEFAULT
        if self.sort_options:
            alg_option = functools.reduce(operator.or_, self.sort_options)
        # FIXME Ambiguity in sorting if we have are reading from multiple directories ?
        return sort_fn(filelist, alg=alg_option)


class FileArrayConfig(FileConfig, arbitrary_types_allowed=True):
    config_type: Literal['array'] = Field(default='array', repr=False)
    dtype: Optional[np.dtype] = None
    shape: Optional[conlist(PositiveInt, min_items=1)] = None

    @validator('format')
    def is_loadable(cls, value, values):
        if value is None:
            format = values['path'].suffix.strip().lower()
            if format not in format_defs.keys():
                raise ValueError('Need a loadable format to load array from file')

    @validator('dtype', pre=True)
    def check_dtype(cls, value):
        """
        Could do this with composition or re-use
        """
        if value is not None:
            try:
                value = np.dtype(value)
            except TypeError:
                raise ValueError(f'Cannot cast {value} to dtype')
        return value

    def resolve(self):
        path = super().resolve()
        # Explicit path needed to avoid recursion problem in FileConfig.load()
        array = self.load(path=path)
        if self.dtype is not None:
            array = array.astype(self.dtype)
        if self.shape is not None:
            if math.prod(self.shape) != array.size:
                raise RuntimeError(
                    'Loaded array does not match config-supplied shape, '
                    f'got array of shape {array.shape} for reshape of {tuple(self.shape)}'
                )
            array = array.reshape(self.shape)
        return array


class InlineArrayConfig(WithRootModel, arbitrary_types_allowed=True):
    config_type: Literal['array'] = Field(default='array', repr=False)
    data: np.ndarray
    dtype: Optional[np.dtype] = None
    shape: Optional[conlist(PositiveInt, min_items=1)] = None

    @validator('data', pre=True)
    def cast_data(cls, value):
        try:
            value = np.asarray(value)
        except TypeError:
            raise ValueError(f'Cannot convert {value} to array')
        return value

    @validator('dtype', pre=True)
    def check_dtype(cls, value):
        if value is not None:
            try:
                value = np.dtype(value)
            except TypeError:
                raise ValueError(f'Cannot cast {value} to dtype')
        return value

    @validator('shape')
    def validate_shape_matches(cls, value, values):
        value = tuple(value)
        shape_size = math.prod(value)
        if 'data' not in values:
            raise ValueError('Cannot get data size')
        if shape_size != values['data'].size:
            raise ValueError('Array shape must have same size as data '
                             f'got {value} for data shape {values["data"].shape}')
        return value

    def resolve(self):
        array = np.asarray(self.data)
        if self.dtype is not None:
            array = array.astype(self.dtype)
        if self.shape is not None:
            array = array.reshape(self.shape)
        return array


class ArrayConfig(WithExtraModel):
    config_type: Literal['array'] = Field(default='array', repr=False)
    array_config: Union[InlineArrayConfig, FileArrayConfig]

    @root_validator(pre=True)
    def wrap_config(cls, values):
        if 'array_config' in values:
            pass
        else:
            values = {
                'config_type': values.get('config_type', 'array'),
                'array_config': values,
            }
        return values

    def resolve(self):
        return self.array_config.resolve()


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
    )(FileConfig.from_value)


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

[my_fileset]
config_type='fileset'
files='yoyo'
sort='natsorted'
sort_options=['FLOAT']

[my_array]
config_type='array'
path = 'test.npy'
dtype='uint8'
shape=[6, 6]
"""
    nest = TreeFactory.from_string(config_str)
    # file_config = freeze_tree(nest['dataset_config'])
    # ds_model = MIBDatasetConfig(**file_config)
    # fileset_model = FileSetConfig(**freeze_tree(nest['my_fileset']))
    array_model = ArrayConfig(**freeze_tree(nest['my_array']))
