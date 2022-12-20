from typing import Dict, Any, Union, Sequence, Optional, List

import numpy as np
import pathlib
import functools
import operator

from libertem.corrections import CorrectionSet

import natsort
import numpy.typing as nt

from config_base import SpecBase, ParserException
from utils import format_defs, sort_methods, sort_types, sort_enum_names, resolve_path_glob


file_schema = {
    "type": "file",
    "title": "FileSpec",
    "properties": {
        "type": {
            "const": "file",
        },
        "path": {
            "type": "string",
        },
        "format": {
            "type": "string",
            "enum": [*format_defs.keys()]
        },
        "load_options": {
            "type": "object",
        },
    },
    "required": ["path"],
}


class FileSpec(SpecBase):
    spec_type = 'file'
    resolve_to = pathlib.Path

    @property
    def path(self) -> pathlib.Path:
        paths = resolve_path_glob(self['path'], self.root)
        if len(paths) != 1:
            raise ParserException(f'path {self["path"]} matched {len(paths)} files')
        return paths[0]

    @property
    def format(self) -> str:
        format = self.get('format', None)
        if format is not None:
            return format.strip().lower()
        return format

    @property
    def load_options(self) -> Dict[str, Any]:
        return self.get('load_options', {})

    def load(self) -> np.ndarray:
        if self.format is None:
            format = self.path.suffix.lstrip('.').lower()
        else:
            format = self.format
        if format not in format_defs.keys():
            raise ParserException(f'Unrecognized file format {format}')
        return format_defs[format](self.path, **self.load_options)

    @classmethod
    def construct(cls, arg, parent=None):
        if isinstance(arg, str):
            instance = cls(path=arg)
            instance._set_parent(parent)
            return instance
        else:
            return super().construct(arg, parent=parent)

    @classmethod
    def validate(cls, checker, instance):
        valid = super().validate(checker, instance)
        valid = valid and ('path' in instance)
        return valid and isinstance(instance['path'], (str, pathlib.Path))

    def resolve(self):
        return self.path


fileset_schema = {
    "type": "fileset",
    "title": "FileSetSpec",
    "properties": {
        "type": {
            "const": "fileset",
        },
        "files": {
            "oneOf": [
                {
                    "type": "string"
                },
                {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
                    "minLength": 1,
                },
            ]
        },
        "sort": {
            "type": "string",
            "enum": [*sort_methods.keys()]
        },
        "sort_options": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": [*sort_enum_names]
            },
        },
    },
    "required": ["files"],
}


class FileSetSpec(SpecBase):
    spec_type = 'fileset'
    resolve_to = List[pathlib.Path]

    @property
    def filelist(self):
        if isinstance(self.files, (str, pathlib.Path)):
            filelist = resolve_path_glob(self.files, self.root)
        elif isinstance(self.files, (list, tuple)):
            # List of (potentially mixed) absolute, relative, or glob specifiers
            filelist = [f for path in self.files for f in resolve_path_glob(path, self.root)]
        else:
            raise ParserException(f'Unrecognized files specifier {self.files}')

        if not filelist:
            raise ParserException(f'Found no files with specifier {self.files}.')

        # It's possible that multiple globs together may match a file more than once
        # Could add some form of uniqueness check for resolved paths ?
        if self.sort:
            filelist = self._sort(filelist)

        return filelist

    @property
    def files(self) -> Union[str, None, Sequence[str]]:
        return self.get('files', None)

    @property
    def sort(self) -> sort_types:
        return self.get('sort', False)

    @property
    def sort_options(self) -> Union[str, Sequence[str]]:
        return self.get('sort_options', False)

    def _sort(self, filelist):
        sort_fn = sort_methods.get(self.sort, None)
        if not sort_fn:
            raise ParserException(f'Unrecognized sort method {self.sort} '
                                  f'options are {tuple(sort_methods.keys())}')
        alg_option = natsort.ns.DEFAULT
        if self.sort_options:
            option_ints = tuple(getattr(natsort.ns, option, None)
                                for option in self.sort_options)
            if None in option_ints:
                invalid_idx = option_ints.index(None)
                invalid_option = self.sort_options[invalid_idx]
                raise ParserException(f'Unrecognized sort option {invalid_option}')
            if len(option_ints) > 1:
                alg_option = functools.reduce(operator.or_, option_ints)
            else:
                alg_option = option_ints[0]

        # FIXME Ambiguity in sorting if we have are reading from multiple directories ?
        return sort_fn(filelist, alg=alg_option)

    @classmethod
    def construct(cls, arg, parent=None):
        if isinstance(arg, (tuple, list, str)):
            instance = cls(files=arg)
            instance._set_parent(parent)
            return instance
        else:
            return super().construct(arg, parent=parent)

    @classmethod
    def validate(cls, checker, instance):
        valid = super().validate(checker, instance)
        valid = valid and ('files' in instance)
        if valid:
            files_val = instance['files']
            if isinstance(files_val, (str, pathlib.Path)):
                pass
            elif isinstance(files_val, (list, tuple)):
                valid = all(isinstance(s, (str, pathlib.Path)) for s in files_val)
        return valid

    def resolve(self):
        return self.filelist


array_schema = {
    "type": "nparray",
    "title": "ArraySpec",
    "properties": {
        "type": {
            "const": "nparray",
        },
        "read_as": {
            "type": "string",
            "enum": ["file"],
        },
        "data": {
            "type": "array",
        },
        "shape": {
            "type": "array",
            "items": {
                "type": "integer",
                "minLength": 1,
            },
        },
        "dtype": {
            "type": "dtype",
        },
    },
    "anyOf": [
        {"required": ["data"]},
        {"required": ["read_as"]},
    ],
}


class ArraySpec(SpecBase):
    spec_type = 'nparray'
    resolve_to = np.ndarray

    @property
    def data(self):
        if 'data' not in self:
            raise ParserException('Require key "data"')
        return self.get('data')

    @property
    def dtype(self) -> Optional['nt.DTypeLike']:
        return self.get('dtype', None)

    @property
    def shape(self) -> Optional[Sequence[int]]:
        return self.get('shape', None)

    @property
    def array(self) -> np.ndarray:
        array = np.asarray(self.data)
        if self.dtype is not None:
            array = array.astype(self.dtype)
        if self.shape is not None:
            array = array.reshape(self.shape)
        return array

    def resolve(self) -> np.ndarray:
        if self.read_as is not None:
            return self.view(self.read_as).resolve()
        return self.array

    @classmethod
    def readers(cls):
        return {
            'file': cls._from_file
        }

    @classmethod
    def construct(cls, arg, parent=None):
        if isinstance(arg, (np.ndarray, list, tuple)):
            instance = cls(data=arg)
            instance._set_parent(parent)
            return instance
        else:
            return super().construct(arg, parent=parent)

    @classmethod
    def validate(cls, checker, instance):
        if instance.read_as is not None:
            view_instance = instance.view(instance.read_as)
            return view_instance.validate(checker, view_instance)
        valid = super().validate(checker, instance)
        valid = valid and ('data' in instance)
        if valid:
            # Very few things can't be cast to np.ndarray
            # especially with dtype=object, but check at least
            # that the cast succeeds (even if unexpected results)
            # This will check that the reshape and dtype params work
            try:
                _ = instance.array
            except Exception:
                valid = False
        return valid

    def _from_file(self) -> np.ndarray:
        file_form: FileSpec = self.view(self.read_as)
        return file_form.load()


corrections_schema = {
    "type": "correctionset",
    "title": "CorrectionSetSpec",
    "properties": {
        "type": {
            "const": "correctionset",
        },
        "dark_frame": {
            "type": "nparray",
        },
        "gain_map": {
            "type": "nparray",
        },
        "excluded_pixels": {
            "type": "array",
            "items": {
                "type": "array",
                "minLength": 1,
                "items": {
                    "type": "integer",
                }
            },
            "minLength": 1,
        },
        "allow_empty": {
            "type": "boolean",
            "default": False,
        }
    },
    "anyOf": [
        {"required": ["dark_frame"]},
        {"required": ["gain_map"]},
        {"required": ["excluded_pixels"]},
    ],
}


class CorrectionSetSpec(SpecBase):
    spec_type = 'correctionset'
    resolve_to = CorrectionSet

    @property
    def allow_empty(self):
        return self.get('allow_empty', None)

    @property
    def dark_frame(self) -> Dict[str, Any]:
        val = self.get('dark_frame', None)
        return self._get_property(val)

    @property
    def gain_map(self) -> Dict[str, Any]:
        val = self.get('gain_map', None)
        return self._get_property(val)

    @property
    def excluded_pixels(self):
        val = self.get('excluded_pixels', None)
        return self._get_property(val)

    def _get_property(self, val):
        if val is None:
            return
        if isinstance(val, SpecBase):
            instance = val
        elif isinstance(val, pathlib.Path):
            instance = FileSpec.construct(val, parent=self.parent)
        elif isinstance(val, str):
            try:
                # try to resolve as a key in tree
                instance = self.resolve_key(val)
            except KeyError:
                # assume it points to a loadable file
                instance = FileSpec.construct(val, parent=self.parent)
        else:
            raise ParserException(f'Unrecognized format: {val}')
        if not isinstance(instance, (FileSpec, ArraySpec)):
            raise ParserException('Only file or nparray supported for correction values')
        if isinstance(instance, FileSpec):
            instance = instance.view('nparray', read_as='file')
        return instance

    @classmethod
    def validate(cls, checker, instance):
        valid = super().validate(checker, instance)
        # FIXME Need to accept dict in case of nested property dicst
        # which aren't necessarily cast to Spec by the time this is called
        # Should perform all casting THEN run all the validate methods
        accepted_types = (str, pathlib.Path, FileSpec, ArraySpec, dict)
        props = {
            'dark_frame': accepted_types,
            'gain_map': accepted_types,
            'excluded_pixels': accepted_types,
            'allow_empty': bool,
        }
        valid = valid and any(d in instance for d in props.keys())
        for prop, types in props.items():
            if prop not in instance:
                continue
            valid = valid and isinstance(instance[prop], types)
        return valid

    def resolve(self) -> CorrectionSet:
        kwargs = {}
        if self.dark_frame is not None:
            kwargs['dark'] = self.dark_frame.resolve()
        if self.gain_map is not None:
            kwargs['gain'] = self.gain_map.resolve()
        if self.excluded_pixels is not None:
            kwargs['excluded_pixels'] = self.excluded_pixels.resolve()
        if self.allow_empty is not None:
            kwargs['allow_empty'] = self.allow_empty
        return CorrectionSet(**kwargs)


class DataSetSpec(SpecBase):
    spec_type = 'dataset'

    @property
    def format(self):
        return self.get('format', None)


roi_schema = {
    "type": "roi",
    "title": "ROISpec",
    "properties": {
        "type": {
            "const": "roi",
        },
        "read_as": {
            "type": "string",
            "enum": ["file", "nparray"],
        },
        "roi_base": {
            "type": "boolean",
        },
        "shape": {
            "type": "array",
            "items": {
                "type": "integer",
                "minLength": 1,
            },
        },
        "dtype": {
            "type": "dtype",
            "default": bool,
        },
        "toggle_px": {
            "type": "array",
            "items": {
                "type": "array",
                "minLength": 1,
                "items": {
                    "type": "integer",
                }
            },
            "minLength": 1,
        },
    },
    "anyOf": [
        {"required": ["roi_base", "toggle_px", "shape"]},
        {"required": ["read_as"]},
    ],
}


class ROISpec(SpecBase):
    spec_type = 'roi'
    resolve_to = np.ndarray

    @property
    def roi_base(self):
        return self['roi_base']

    @property
    def shape(self):
        return self['shape']

    @property
    def dtype(self):
        return self.get('dtype', bool)

    @property
    def toggle_px(self):
        # Existence and type checked in validate
        return self['toggle_px']

    @property
    def array(self):
        array = np.full(self.shape, self.roi_base, dtype=self.dtype)
        for toggle in self.toggle_px:
            array[tuple(toggle)] = not self.roi_base
        return array

    def resolve(self) -> np.ndarray:
        if self.read_as is not None:
            return self.view(self.read_as).resolve()
        return self.array

    @classmethod
    def readers(cls):
        return {
            'file': cls._from_file,
            'nparray': cls._from_array,
        }

    @classmethod
    def validate(cls, checker, instance):
        if instance.read_as is not None:
            view_instance = instance.view(instance.read_as)
            return view_instance.validate(checker, view_instance)
        valid = super().validate(checker, instance)
        required = {
            'shape': (list, tuple, np.ndarray),
            'roi_base': (bool,),
            'toggle_px': (list, tuple, np.ndarray),
        }
        for key, types in required.items():
            valid = valid and key in instance and isinstance(instance[key], types)
        valid = valid and ('data' in instance)
        return valid

    def _from_file(self):
        file_form = self.view(self.read_as)
        return file_form.load()

    def _from_array(self):
        array_form = self.view(self.read_as)
        return array_form.resolve()


# class ContextSpec(SpecBase):
#     spec_type = 'context'


# class AnalysisSpec(SpecBase):
#     spec_type = 'analysis'


# class UDFSpec(SpecBase):
#     spec_type = 'udf'


# class RunSpec(SpecBase):
#     spec_type = 'run'
