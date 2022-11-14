from typing import Dict, Any, Union, Sequence, Optional, List, TYPE_CHECKING, NamedTuple
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


if TYPE_CHECKING:
    import numpy.typing as nt


sort_types = Literal['natsorted', 'humansorted', 'os_sorted']
enum_names = tuple(en.name for en in natsort.ns)
spec_type_key = 'type'

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
    if not isinstance(dotpath, str):
        raise TypeError(f'Cannot resolve key {dotpath}')
    components = dotpath.split('.')
    view = struct
    for c in components:
        if c not in view:
            raise TypeError(f'Cannot resolve key {dotpath}')
        view = view.get(c)
    return view


def dotpath_exists(struct: Dict, dotpath: str):
    try:
        _ = resolve_dotpath(struct, dotpath)
        return True
    except TypeError:
        return False


class SpecBase(dict):
    spec_type = 'base'
    required_keys = []
    parse_as = None

    def __init__(self, **spec):
        super().__init__(**spec)
        self._root_structure = self
        self.validate()

    def validate(self):
        # Add read_as injection of required_keys here
        # consider needing some requires from type and some from read_as
        missing = tuple(k for k in self.required_keys if k not in self)
        if missing:
            raise TypeError(f'Missing keys {missing} for {self.spec_type}')

    def parse(self, parse_as: Optional[NamedTuple] = None):
        # Convert self into a NamedTuple of fields specified in
        # either the parse_as argument or the default parse_as for the class
        if parse_as is None:
            if self.parse_as is None:
                raise TypeError(f'Cannot parse {self.__class__.__name__} '
                                'to object without type definition')
            parse_as = self.parse_as
        fields = {}
        for field in parse_as._fields:
            try:
                fields[field] = getattr(self, field)
            except AttributeError as e:
                if field not in parse_as._field_defaults:
                    raise e
        return parse_as(**fields)

    def load(self):
        # Try to load the oject defined by this spec
        # Will call load on all sub-specs (assumed to be required)
        raise NotImplementedError('Cannot load a bare SpecBase')

    @property
    def root(self) -> pathlib.Path:
        return self.get('root', None)

    def read_as(self) -> Optional[str]:
        return self.get('read_as', None)

    def _set_root_structure(self, struct: Dict[str, Any]):
        self._root_structure = struct

    def resolve(self, key):
        """
        Get key from self in dot notation (a.b.c)
        then try to get key from the root tree
        If not available then raise
        """
        try:
            return resolve_dotpath(self, key)
        except AttributeError as e:
            if self._root_structure is not self:
                return resolve_dotpath(self._root_structure, key)
            raise e

    def as_tree(self, level=0, name=None):
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
            return '\n'.join(lines)
        else:
            return lines

    def __repr__(self):
        return f'{self.__class__.__name__}({super().__repr__()})'


class FileT(NamedTuple):
    path: pathlib.Path


class FileSpec(SpecBase):
    spec_type = 'file'
    required_keys = ['file']
    parse_as = FileT

    @property
    def file(self) -> Optional[str]:
        return self.get('file', None)

    @property
    def path(self) -> pathlib.Path:
        file = pathlib.Path(self.file)
        if file.is_absolute():
            return file
        else:
            return (self.root / file).resolve()

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


class FileSetT(NamedTuple):
    filelist: List[pathlib.Path]


class FileSetSpec(SpecBase):
    spec_type = 'fileset'
    required_keys = ['files']
    parse_as = FileSetT

    @property
    def filelist(self):
        if isinstance(self.files, str):
            # files could be glob or a single filename
            # specifying a list of files
            filepath = pathlib.Path(self.files)
            if not filepath.is_absolute():
                filepath = self.root / filepath
            if filepath.is_file():
                # assume this is a textfile of files to read
                # FIXME Catch which exceptions if assumption wrong ?
                with filepath.open('r') as fp:
                    filelist = filepath.readlines()
                # Implicitly paths relative to text file specifier
                filelist = [(filepath.parent / f).resolve() for f in filelist]
            else:
                # assume glob
                filelist = glob.glob(filepath)
                filelist = [pathlib.Path(f).resolve() for f in filelist]
        elif isinstance(self.files, list):
            # List of (potentially mixed) absolute, relative, or glob specifiers
            filelist = [pathlib.Path(f) for f in self.files]
            filelist = [f if f.is_absolute() else (self.root / f) for f in filelist]
            # Need to test if is glob here else we get errors too early for
            # absolute files which are not present on the FS
            filelist = [ff for f in filelist for ff in glob.glob(str(f))]
            filelist = [pathlib.Path(f).resolve() for f in filelist]
        else:
            raise ParserException(f'Unrecognized files specifier {type(self.files)}')

        if not filelist:
            raise ParserException(f'Found no files with specifier {self.files}.')

        if self.sort:
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
            filelist = sort_fn(filelist, alg=alg_option)

        # Could add existence checks here but the dataset should already do this!
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


class ArrayT(NamedTuple):
    array: np.ndarray


class ArraySpec(SpecBase):
    spec_type = 'array'
    required_keys = ['data']
    parse_as = ArrayT

    @property
    def raw_data(self) -> List:
        return self.get('data', None)

    @property
    def dtype(self) -> Optional['nt.DTypeLike']:
        return self.get('dtype', None)

    @property
    def shape(self) -> Optional[Sequence[int]]:
        return self.get('shape', None)

    @property
    def array(self) -> np.ndarray:
        array = np.asarray(self.raw_data)
        if self.dtype is not None:
            array = array.astype(self.dtype)
        if self.shape is not None:
            array = array.reshape(self.shape)
        return array

    def load(self) -> np.ndarray:
        return self.array


class MaskSpec(SpecBase):
    spec_type = 'mask'


class CorrectionSetSpec(SpecBase):
    spec_type = 'correctionset'
    optional_keys = ['dark_frame', 'gain_map', 'excluded_pixels']

    def validate(self):
        if not any(k in self for k in self.optional_keys):
            raise ParserException("Correction set doesn't define any known corrections")

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


class ROISpec(SpecBase):
    spec_type = 'roi'
    # default read_as ?


class ContextSpec(SpecBase):
    spec_type = 'context'


class AnalysisSpec(SpecBase):
    spec_type = 'analysis'


class UDFSpec(SpecBase):
    spec_type = 'udf'


class RunSpec(SpecBase):
    spec_type = 'run'


parsers = {
    'file': FileSpec,
    'fileset': FileSetSpec,
    'array': ArraySpec,
    'dataset': DataSetSpec,
    'correctionset': CorrectionSetSpec,
    'roi': ROISpec,
    'context': ContextSpec,
    'analysis': AnalysisSpec,
    'udf': UDFSpec,
    'run': RunSpec,
}


def parse_spec(struct: Dict[str, Any]):
    if not isinstance(struct, dict):
        return struct
    for key, value in struct.items():
        if isinstance(value, dict):
            if 'root' not in value and 'root' in struct:
                struct[key] = parse_spec({**value, 'root': struct['root']})
            else:
                struct[key] = parse_spec(value)
    if spec_type_key in struct:
        parser = parsers.get(struct[spec_type_key], None)
        if parser is None:
            raise TypeError(f'Unrecognized spec type {struct[spec_type_key]}')
        return parser(**struct)
    else:
        return struct


class SpecTree(SpecBase):
    def __init__(self, **spec):
        super().__init__(**spec)
        self.set_root(self, self)

    @staticmethod
    def set_root(struct, root):
        for value in struct.values():
            if isinstance(value, SpecBase):
                value._set_root_structure(root)
            if isinstance(value, dict):
                SpecTree.set_root(value, root)

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

        return cls(**struct)


if __name__ == '__main__':
    nest = SpecTree.from_file('./sidecar.toml')

    class NullFileT(NamedTuple):
        path: pathlib.Path
        dtype: np.typing.DTypeLike = np.float32


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
