from typing import (Dict, Any, Union, Sequence, Optional, List,
                    TYPE_CHECKING, NamedTuple, Type, Tuple)
from typing_extensions import Literal
import glob
import numpy as np
import pathlib
import tomli
import json
import functools
import operator
from skimage.io import imread

from libertem.corrections import CorrectionSet

import natsort
import numpy.typing as nt


sort_types = Literal['natsorted', 'humansorted', 'os_sorted']
enum_names = tuple(en.name for en in natsort.ns)

class ParserException(Exception):
    ...


def load_raw(path, *, shape, dtype):
    return np.fromfile(path, dtype=dtype).reshape(shape)


def load_image(path, **kwargs):
    return imread(path, **kwargs)


format_defs = {
    'raw': load_raw,
    'bin': load_raw,
    'npy': np.load,
    'tiff': load_image,
    'tif': load_image,
    'jpg': load_image,
    'png': load_image,
}


sort_methods = {
    'natsorted': natsort.natsorted,
    'os_sorted': natsort.os_sorted,
    'humansorted': natsort.humansorted,
}


def resolve_dotpath(struct: Dict, dotpath: str):
    return _resolve_generic(struct, dotpath, '.', '.')


def resolve_jsonpath(struct: Dict, jsonpath: str):
    return _resolve_generic(struct, jsonpath, '#/', '/')


def _resolve_generic(struct: Dict, path: str, strip: str, split: str):
    if not isinstance(path, str):
        raise TypeError(f'Cannot resolve key {path}')
    components = path.strip().strip(strip).split(split)
    view = struct
    for c in components:
        if not isinstance(view, dict) or c not in view:
            raise TypeError(f'Cannot resolve key {path}')
        view = view.get(c)
    return view


def dotpath_exists(struct: Dict, dotpath: str):
    try:
        _ = resolve_dotpath(struct, dotpath)
        return True
    except TypeError:
        return False


def find_tree_root(struct: 'NestedDict'):
    if struct.parent is None:
        return struct
    return find_tree_root(struct.parent)


class NestedDict(dict):
    def _set_parent(self, parent: Dict[str, Any]):
        self._parent = parent

    @property
    def parent(self):
        try:
            return self._parent
        except AttributeError:
            return None

    def resolve_key(self, key: str):
        """
        Get key from tree in JSON path notation
           i.e. #/key1/key2
        starting from root or if a bare key is provided
        resolve from self (using dot notation)
        then try to get key from the root tree
        If not available then raise
        """
        if not isinstance(key, str):
            raise TypeError(f'Cannot resolve key {key}')
        if key.startswith('#'):
            root = find_tree_root(self)
            return resolve_jsonpath(root, key)
        else:
            return resolve_dotpath(self, key)

    @property
    def root(self) -> Optional[pathlib.Path]:
        root = self.get('root', None)
        if root is None and self.parent is not None:
            return self.parent.root
        else:
            try:
                return pathlib.Path(root)
            except TypeError:
                return root


class SpecBase(NestedDict):
    spec_type = 'base'

    def __init__(self, **kvals):
        super().__init__(self, **kvals)
        if not self.validate(None, self):
            raise ParserException(f'Invalid spec for {self}')

    def load(self):
        # Try to load the oject defined by this spec
        # Will call load on all sub-specs (assumed to be required)
        raise NotImplementedError(f'No load method for {self.__class__.__name__}')

    def as_tree(self, level=0, name=None, do_print=True):
        ident = '  ' * level + f'{self.__class__.__name__}'
        if name is not None:
            ident = ident + f'   [{name}]'
        lines = [ident]
        if name is not None:
            lines[-1]
        for key, value in self.items():
            if isinstance(value, SpecBase):
                lines.extend(value.as_tree(level=level + 1, name=key))
        if level == 0:
            tree_string = '\n'.join(lines)
            if do_print:
                print(tree_string)
            else:
                return tree_string
        else:
            return lines

    def __repr__(self):
        return f'{self.__class__.__name__}({super().__repr__()})'

    @classmethod
    def construct(cls, arg, parent=None):
        if isinstance(arg, dict):
            instance = cls(**arg)
            # Retain arg's parent while casting if it
            # is already a NestedDict, even if None,
            # else use the external parent
            try:
                instance._set_parent(arg.parent)
            except AttributeError:
                instance._set_parent(parent)
            return instance
        else:
            raise ParserException(f'Unrecognized spec {arg} for {cls.__name__}')

    @classmethod
    def validate(cls, checker, instance):
        if instance.read_as is not None and instance.read_as not in instance.readers():
            return False
        return isinstance(instance, cls)

    @property
    def read_as(self):
        return self.get('read_as', None)

    def view(self, spec_type: str):
        """
        Get a copy of this instance as another type

        Removes the read_as key from the copy
        and sets 'type' to equal the new type
        """
        if spec_type not in parsers:
            raise ParserException(f'Cannot view {self.__class__.__name__} as {spec_type}')
        instance_props = {k: v for k, v in self.items() if k != 'read_as'}
        if 'type' in instance_props:
            instance_props['type'] = spec_type
        instance = parsers[spec_type](**instance_props)
        instance._set_parent(self.parent)
        return instance

    @classmethod
    def readers(cls):
        return {}

    def resolve(self):
        raise NotImplementedError('Cannot reosolve bare SpecBase')

    def resolve_as(self, spec_type):
        if spec_type in self.readers():
            return self.readers()[self.read_as](self)
        raise ParserException(f'Unrecognized read_as "{self.read_as}" for '
                              f'{self.__class__.__name__}')


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


def resolve_path_glob(path: pathlib.Path, root_dir: pathlib.Path) -> List[pathlib.Path]:
    """
    Resolve path using glob expansion
    If path is relative concatenate path with root_dir.
    A path containing no glob characters
    will still resolve in glob.glob to List[path]
    Return fully resolved paths

    This essentially performs an existence check for files
    Might not be desirable for cases where files are not present
    on the main node but are on other nodes ?
    """
    filepath = pathlib.Path(path).expanduser()
    if not filepath.is_absolute():
        filepath = root_dir / filepath
    return [pathlib.Path(p).resolve() for p in glob.glob(str(filepath))]


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
            return self.resolve_as(self.read_as)
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

    def _from_file(self):
        file_form = self.view(self.read_as)
        return file_form.load()


class CorrectionSetSpec(SpecBase):
    spec_type = 'correctionset'
    resolve_to = CorrectionSet

    @property
    def dark_frame(self) -> Dict[str, Any]:
        return self.get('dark_frame', None)

    @property
    def gain_map(self) -> Dict[str, Any]:
        return self.get('gain_map', None)

    @property
    def excluded_pixels(self):
        return self.get('excluded_pixels', None)

    def load(self):
        return CorrectionSet()


class DataSetSpec(SpecBase):
    spec_type = 'dataset'

    @property
    def format(self):
        return self.get('format', None)


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
            return self.resolve_as(self.read_as)
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


spec_types = [
    FileSpec,
    FileSetSpec,
    ArraySpec,
    DataSetSpec,
    CorrectionSetSpec,
    ROISpec,
]
parsers = {s.spec_type: s for s in spec_types}


def parse_spec(struct: Dict[str, Any], parent=None):
    if not isinstance(struct, dict):
        return struct
    if 'type' in struct:
        parser = parsers.get(struct['type'], None)
        if parser is None:
            raise TypeError(f'Unrecognized spec type: "{struct["type"]}"')
        struct = parser(**struct)
    elif parent is None:
        # Must be root node
        struct = SpecTree(**struct)
    else:
        # Convert regular dict to a NestedDict so it can reference parent
        struct = NestedDict(**struct)
    struct._set_parent(parent)
    for key, value in struct.items():
        if isinstance(value, dict):
            struct[key] = parse_spec(value, parent=struct)
    return struct


class SpecTree(SpecBase):
    @classmethod
    def from_file(cls, path):
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

        if 'root' not in struct:
            # set parent directory of config file
            struct['root'] = path.parent

        return cls._get_tree(struct)

    @classmethod
    def from_string(cls, string, format='toml'):
        if format == 'toml':
            struct = tomli.loads(string)
        elif format == 'json':
            struct = json.loads(string)
        else:
            raise ParserException(f"Unrecognized format {format}")

        if 'root' not in struct:
            # set cwd
            struct['root'] = pathlib.Path()

        return cls._get_tree(struct)

    @classmethod
    def _get_tree(cls, struct: Dict[str, Any]):
        return parse_spec(struct)


class WrappedType:
    @classmethod
    def validate(cls, checker, instance):
        raise NotImplementedError()

    @classmethod
    def construct(cls, arg, parent=None):
        return arg


class DType(WrappedType):
    spec_type = 'dtype'

    @classmethod
    def validate(cls, checker, instance):
        try:
            cls.construct(instance)
            return True
        except TypeError:
            return False

    @classmethod
    def construct(cls, arg, parent=None):
        dtype = np.dtype(arg)
        if dtype.type is not None:
            return dtype.type
        return dtype


wrapped_types = (DType,)
types = {
    **parsers,
    **{t.spec_type: t for t in wrapped_types},
}

if __name__ == '__main__':
    nest = SpecTree.from_file('./sidecar_file.toml')


# ds = ctx.load('ds_def.toml')  # with/without 'auto' key ??
# ds = ctx.load('raw', 'ds_def.toml')
# ds = RawDataSet('ds_def.toml')  # problematic if already instantiated
# ds = RawDataSet('ds_def.toml', nav_shape=(50, 60))  # hybrid (what priority?)
# ds = RawDataSet.from_spec('ds_def.toml')

# REQUIRE
# all ds args/kwargs are optional except first arg
# first argument can be:
# -> a path pointing to the data itself
#    (or any other first arg/kwarg the DataSet constructor normally accepts)
# -> a path pointing to markup containing a single (matching) dataset def
# -> an already-parsed dataset descriptor for that DataSet
# -> a dict to convert into a ds descriptor for that DataSet
# No direct correspondence between __init__ of DataSet and descriptor
# The descriptor is read during initialize() using methods on the dataset
# The Dataset is responsible for interpreting the content of the descriptor
# but the parsing / loading of any supporing parameters/data is provided
# by the Spec system
# This way datasets can store references to CorrectionSet (for example)
# without requiring `corrections=None` in __init__
# an initialized DataSetSpec can have a .load(ctx) method which
# knows how to call ctx.load(self) to return the dataset object
# (even use an Inline ctx for loading if not supplied)
# Spec should have access to the full top-level spec so it can reference
# keys outside of itself for accessing shared file/set descriptions
# potential name collision if keys are external ??
