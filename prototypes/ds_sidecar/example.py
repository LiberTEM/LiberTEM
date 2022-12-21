import tempfile
import numpy as np
import pathlib
from jsonschema.exceptions import ValidationError

import libertem.api as lt
from libertem.io.dataset.raw import RawFileDataSet

from specs import DataSetSpec
from config_base import NestedDict
from utils import ParserException
from spec_tree import SpecTree


raw_ds_schema = {
    "type": "dataset",
    "title": "RAW dataset",
    "properties": {
        "path": {
            "type": "file",
        },
        "format": {
            "enum": ["raw", "RAW"],
        },
        "nav_shape": {
            "$ref": "#/$defs/shape",
        },
        "sig_shape": {
            "$ref": "#/$defs/shape",
        },
        "dtype": {
            "type": "dtype",
        },
        "sync_offset": {
            "type": "integer",
            "default": 0,
        },
    },
    "required": ["path", "nav_shape", "sig_shape", "dtype"],
    "$defs": {
        "shape": {
            "type": "array",
            "items": {
                "type": "integer",
                "minimum": 1
            },
            "minItems": 1,
        }
    }
}

class RawFileDataSetConfig(RawFileDataSet):
    """
    Subclass RawFileDataSet to add config file support
    """
    def initialize(self, executor):
        if is_config_def(self._path):
            ds_config = load_ds_config_with_schema(self._path, raw_ds_schema)
            self._path = ds_config['path'].resolve()
            self._dtype = ds_config['dtype']
            self._nav_shape = tuple(ds_config['nav_shape'])
            self._sig_shape = tuple(ds_config['sig_shape'])
            self._sync_offset = ds_config['sync_offset']
        return super().initialize(executor)


def is_config_def(value) -> bool:
    if isinstance(value, dict):
        return True
    elif isinstance(value, (str, pathlib.Path)):
        value = pathlib.Path(value)
        if value.is_file() and value.suffix in ('.toml', '.json'):
            return True
    return False


def load_ds_config_with_schema(config, schema):
    if isinstance(config, dict):
        nest = SpecTree.to_tree(config)
    else:
        config_path = pathlib.Path(config)
        nest = SpecTree.from_file(config_path)

    def check_sub_configs(value):
        try:
            # The validator checks the top-level type
            # so we have to give it a DataSetSpec if it is ever going to pass
            if isinstance(value, NestedDict) and not isinstance(value, DataSetSpec):
                value = DataSetSpec.construct(value)
            _ = value.apply_schema(schema)
            return True
        except (ValidationError, ParserException):
            return False

    ds_configs = tuple(nest.search(check_sub_configs))
    if not ds_configs:
        raise ParserException('No matching definitions for dataset')
    elif len(ds_configs) > 1:
        raise ParserException('Multiple matching definitions for dataset')
    return DataSetSpec.construct(ds_configs[0]).apply_schema(schema)


if __name__ == '__main__':
    nav_shape = (8, 8)
    sig_shape = (16, 16)
    dtype = np.float32
    data: np.ndarray = np.random.uniform(size=nav_shape + sig_shape).astype(dtype)

    with tempfile.TemporaryDirectory() as td:
        tempdir = pathlib.Path(td)
        filepath = tempdir / 'data.raw'
        data.tofile(filepath)

        toml_def = f"""format = 'raw'
path = '{filepath}'
dtype = '{np.dtype(dtype)}'
nav_shape = {list(nav_shape)}
sig_shape = {list(sig_shape)}"""

        toml_path = tempdir / 'config.toml'
        with toml_path.open('w') as fp:
            fp.write(toml_def)

        # Required to provide nav/sig shape for this dataset, would need to change it!
        ds = RawFileDataSetConfig(toml_path, None, nav_shape=(1, 1), sig_shape=(1, 1))
        ctx = lt.Context.make_with('inline')
        ds.initialize(ctx.executor)
