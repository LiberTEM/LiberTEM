import pathlib
import tomli
import json

from typing import Dict, Any

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


class SpecTree(SpecBase):
    spec_type = 'tree'

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

        return cls.to_tree(struct)

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

        return cls.to_tree(struct)

    @classmethod
    def to_tree(cls, struct: Dict[str, Any]):
        return parse_spec(struct)


def parse_spec(struct: Dict[str, Any], parent=None):
    if not isinstance(struct, dict):
        return struct
    if 'type' in struct:
        parser = spec_types.get(struct['type'], None)
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
