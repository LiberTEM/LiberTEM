import pathlib
import math
import numpy as np
import natsort
import functools
import operator

from typing import Dict, Any, Optional, Union, List
from typing_extensions import Literal

from pydantic import BaseModel, Extra, validator, root_validator
from pydantic import conlist, PositiveInt, Field

from utils import resolve_path_glob, join_if_relative
from utils import format_defs, sort_methods, format_T


class DType:
    """Acts as type annotation/validator for np.dtype-like"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        try:
            value = np.dtype(value)
        except TypeError:
            raise ValueError(f'Cannot cast {value} to dtype')
        return value


class NPArray:
    """Acts as type annotation/validator for np.ndarray-like"""
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, value):
        try:
            value = np.asarray(value)
        except TypeError:
            raise ValueError(f'Cannot convert {value} to array')
        return value


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
        search_path = join_if_relative(self.path, self.root)
        paths = resolve_path_glob(search_path)
        if len(paths) > 1:
            raise RuntimeError(f'Path {search_path} matched {len(paths)} '
                               'files, must match a single file')
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
            path = join_if_relative(self.files, self.root)
            filelist = resolve_path_glob(path)
        elif isinstance(self.files, (list, tuple)):
            # List of (potentially mixed) absolute, relative, or glob specifiers
            paths = [join_if_relative(path, self.root) for path in self.files]
            filelist = [f for path in paths for f in resolve_path_glob(path)]
        else:
            raise ValueError(f'Unrecognized files specifier {self.files}')

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


class FileArrayConfig(FileConfig):
    config_type: Literal['array'] = Field(default='array', repr=False)
    dtype: Optional[DType] = None
    shape: Optional[conlist(PositiveInt, min_items=1)] = None

    @validator('format')
    def is_loadable(cls, value, values):
        if value is None:
            format = values['path'].suffix.strip().lower()
            if format not in format_defs.keys():
                raise ValueError('Need a loadable format to load array from file')

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


class InlineArrayConfig(WithRootModel):
    config_type: Literal['array'] = Field(default='array', repr=False)
    data: NPArray
    dtype: Optional[DType] = None
    shape: Optional[conlist(PositiveInt, min_items=1)] = None

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


class StandardDatasetConfig(WithRootModel):
    config_type: Literal['dataset'] = Field(default='dataset', repr=False)
    # This could be an enum of defined dataset keys
    ds_format: Optional[str] = 'auto'
    path: Union[FileConfig, pathlib.Path, str]
    nav_shape: Optional[conlist(PositiveInt, min_items=1)] = None
    sig_shape: Optional[conlist(PositiveInt, min_items=1)] = None
    sync_offset: Optional[int] = 0

    _cast_file = validator(
        'path',
        pre=True,
        allow_reuse=True
    )(FileConfig.from_value)


if __name__ == '__main__':
    from tree import TreeFactory

    config_str = R"""
root='/home/mat/Data/ds1'

[dataset_config]
config_type='dataset'
ds_format='mib'
path='#/my_mib_file'

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
data = [5, 6, 7, 8]
dtype='uint8'
shape=[2, 2]
"""
    nest = TreeFactory.from_string(config_str)
    ds_model = StandardDatasetConfig(**nest['dataset_config'].freeze())
    fileset_model = FileSetConfig(**nest['my_fileset'].freeze())
    array_model = ArrayConfig(**nest['my_array'].freeze())
