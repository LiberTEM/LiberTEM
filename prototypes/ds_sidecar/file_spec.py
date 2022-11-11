from typing import Dict, Any, Union, Sequence, Optional, List, TYPE_CHECKING
from typing_extensions import Literal
import glob
import numpy as np
import pathlib
import toml
import json
import functools
import operator
from skimage.io import imread

from libertem.corrections import CorrectionSet

import natsort


if TYPE_CHECKING:
    import numpy.typing as nt


sort_types = Literal[False, 'natsorted', 'humansorted', 'os_sorted']
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


class SpecBase:
    spec_type = 'base'
    reserved_keys = ['type']

    def __init__(
        self,
        spec: Dict[str, Any],
        external_root: pathlib.Path,
    ):
        self._type = spec.pop('type', self.spec_type)
        self._spec = spec
        self._external_root = external_root

    @property
    def spec(self) -> Dict[str, Any]:
        return self._spec

    @property
    def root(self) -> pathlib.Path:
        local_root = self.spec.get('root', None)
        if local_root is not None:
            return pathlib.Path(local_root)
        else:
            return self._external_root


class FileSpec(SpecBase):
    spec_type = 'file'
    reserved_keys = ['type', 'file', 'format']

    def __init__(
        self,
        spec: Dict[str, Any],
        external_root: pathlib.Path,
    ):
        super().__init__(spec, external_root)
        if self.file is None:
            raise ParserException('No file path supplied')

    @property
    def file(self) -> Optional[str]:
        return self.spec.get('file', None)

    @property
    def path(self) -> pathlib.Path:
        file = pathlib.Path(self.file)
        if file.is_absolute():
            return file
        else:
            return (self.root / file).resolve()

    @property
    def format(self) -> str:
        # could infer format from file suffix ??
        format = self.spec.get('format', None)
        if format is not None:
            return format.strip().lower()
        return format

    @property
    def other_params(self) -> Dict[str, Any]:
        return {k: v for k, v
                in self.spec.values()
                if k not in self.reserved_keys}

    def load(self) -> np.ndarray:
        if self.format is None:
            return
        if self.format not in format_defs.keys():
            raise ParserException(f'Unrecognized file format {self.format}')
        return format_defs[self.format](self.path, **self.other_params)


class FileSetSpec(SpecBase):
    spec_type = 'fileset'
    reserved_keys = ['type', 'files', 'sort', 'sort_keys']

    def __init__(
        self,
        spec: Dict[str, Any],
        external_root: pathlib.Path,
    ):
        super().__init__(spec, external_root)
        if self.files is None:
            raise ParserException('No files specifier')

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
            filelist = [ff for f in filelist for ff in glob.glob(f)]
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
        self._filelist = filelist

    @property
    def files(self) -> Union[str, None, Sequence[str]]:
        return self.spec.get('files', None)

    @property
    def sort(self) -> sort_types:
        return self.spec.get('sort', False)

    @property
    def sort_options(self) -> Union[False, str, Sequence[str]]:
        return self.spec.get('sort_options', False)

    @property
    def filelist(self) -> List[pathlib.Path]:
        return self._filelist


class ArraySpec(SpecBase):
    spec_type = 'array'
    reserved_keys = ['type', 'data', 'dtype', 'shape']

    def __init__(
        self,
        spec: Dict[str, Any],
        external_root: pathlib.Path,
    ):
        super().__init__(spec, external_root)

        if self.data is None:
            raise ParserException('No data specified in ArraySpec')

    @property
    def data(self) -> List:
        return self.spec.get('data', None)

    @property
    def dtype(self) -> Optional['nt.DTypeLike']:
        return self.spec.get('dtype', None)

    @property
    def shape(self) -> Optional[Sequence[int]]:
        return self.spec.get('shape', None)

    def load(self) -> np.ndarray:
        array = np.asarray(self.data)
        if self.dtype is not None:
            array = array.astype(self.dtype)
        if self.shape is not None:
            array = array.reshape(self.shape)
        return array


class MaskSpec(SpecBase):
    spec_type = 'mask'


class CorrectionSetSpec(SpecBase):
    spec_type = 'correctionset'
    reserved_keys = ['type', 'dark_frame', 'gain_map', 'excluded_pixels', 'allow_empty']

    def __init__(
        self,
        spec: Dict[str, Any],
        external_root: pathlib.Path,
    ):
        super().__init__(spec, external_root)
        # type has been removed already
        if not any(k in self.spec for k in self.reserved_keys):
            raise ParserException("Correction set doesn't define any known corrections")

    @property
    def dark_frame(self) -> Dict[str, Any]:
        return self.spec.get('dark_frame', None)

    @property
    def gain_map(self) -> Dict[str, Any]:
        return self.spec.get('gain_map', None)

    @property
    def excluded_pixels(self):
        return self.spec.get('excluded_pixels', None)

    def load(self):
        CorrectionSet()


class DataSetSpec(SpecBase):
    spec_type = 'dataset'


parsers = {
    'file': FileSpec,
    'fileset': FileSetSpec,
    'array': ArraySpec,
    'dataset': DataSetSpec,
    'correctionset': CorrectionSetSpec,
}


class RootSpec:
    def __init__(self, spec):
        ...

    def files():
        ...

    def datasets():
        ...
    
    def corrections():
        ...

    def contexts():
        ...
    
    def analyses():
        ...


class SpecParser:
    def __init__(self, spec):
        self._structure = self._parse_spec(spec)

    @classmethod
    def from_file(cls, path):
        path = pathlib.Path(path)

        if not path.is_file():
            raise ParserException(f"Cannot find spec file {path}")

        if path.suffix == '.toml':
            struct = toml.load(path)
        elif path.suffix == '.json':
            with path.open('r') as fp:
                struct = json.load(fp)
        # elif path.suffix == '.yaml':
        #     ...
        else:
            raise ParserException(f"Unrecognized format {path.suffix}")

        return SpecParser(struct)

    @classmethod
    def _parse_spec(cls, spec: Dict[str, Any], parse_as: SpecBase):

        if not isinstance(spec, dict):
            raise ParserException(f"Cannot parse non-dict spec, got {type(spec)}")

        structure = {}
        for key, value in spec.items():
            if isinstance(value, dict) and 'type' in value:
                # sub spec
                value_type = value['type']
                parser = parsers.get(value_type, None)
                if parser is None:
                    pass





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
