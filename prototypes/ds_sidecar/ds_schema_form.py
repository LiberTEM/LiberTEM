import pathlib
import numpy as np
from jsonschema import validators

from file_spec import SpecTree


toml_def = """
root = '~/Data/'

[my_metadata_file]
type = 'file'
path = "/home/alex/Data/TVIPS/rec_20200623_080237_000.tvips"
format = 'RAW'
dtype = 'float32'
shape = [64, 64]

[my_data_fileset]
type='fileset'
files = [
    "test.raw",
    "test2.raw"
]
sort = false
sort_options = []

[my_tvips_dataset]
type = "dataset"
format = "tvips"
meta = 'my_metadata_file'
data = 'my_data_fileset'
nav_shape = [32, 32]
sig_shape = [32, 32]
dtype = 'float32'
sync_offset = 0
"""

tvips_schema = {
    "type": "dataset",
    "title": "TVIPS dataset",
    "properties": {
        "meta": {
            "type": "file",
            # file['file'].dtype should be
            # auto-interpreted as format['dtype']
            # "properties": {
            #     "dtype": {
            #         "type": "$dtype"
            #     }
            # },
            "required": ["dtype"],
        },
        "data": {
            "type": "fileset",
        },
        "nav_shape": {
            "$ref": "#/$defs/shape",
        },
        "sig_shape": {
            "$ref": "#/$defs/shape",
        },
        "dtype": {
            "type": "string",
            "format": "dtype",
            "default": "float32"
        },
        "sync_offset": {
            "type": "integer",
            "default": 0,
        },
    },
    "required": ["meta"],
    "$defs": {
        "shape": {
            "type": "array",
            "items": {
                "type": "number",
                "minimum": 1
            },
            "minItems": 1,
        }
    }
}


def is_file(checker, instance):
    return isinstance(instance, (str, pathlib.Path))


def is_fileset(checker, instance):
    if isinstance(instance, (str, pathlib.Path)):
        return True
    elif isinstance(instance, (list, tuple)):
        return all(isinstance(s, (str, pathlib.Path)) for s in instance)
    else:
        return False


def is_dtype(checker, instance):
    try:
        _ = np.dtype(instance)
        return True
    except TypeError:
        return False


def is_dataset(checker, instance):
    return True


"""
Parser tools have builtin conversions to resolve paths,
fill globs, sort, find loaders and create dtypes.
They should validate the schema 'required' args and types
only when the object resolves them using its schema.
They don't validate the parameters themselves, leave that
to the datasets or other objects.

Ideally the parser should return an object which can be
accessed and resolved more than once (on different machines ?)
so that globbing / resolving can happen on the target machine
and not just locally, in the future this could lead to a change
in how we .initialize a dataset
"""


if __name__ == '__main__':
    nest = SpecTree.from_string(toml_def)

    definitions = {
        'file': is_file,
        'dtype': is_dtype,
        'fileset': is_fileset,
        'dataset': is_dataset,
    }

    type_checker = validators.Draft202012Validator.TYPE_CHECKER.redefine_many(
        definitions=definitions
    )

    CustomValidator = validators.extend(
        validators.Draft202012Validator,
        type_checker=type_checker,
    )

    validator = CustomValidator(schema=tvips_schema)
    print(validator.is_valid(nest['my_tvips_dataset']))
