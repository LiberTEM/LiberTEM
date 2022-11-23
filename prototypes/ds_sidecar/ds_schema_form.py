import pathlib
import numpy as np
from functools import partial
from jsonschema import validators
from jsonschema.exceptions import ValidationError

from file_spec import SpecTree, NestedDict, parsers


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
meta = '#/my_metadata_file'
data = '#/my_data_fileset'
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

definitions = {
    'file': is_file,
    'dtype': is_dtype,
    'fileset': is_fileset,
    'dataset': is_dataset,
}


def extend_check_required(validator_class):
    validate_required = validator_class.VALIDATORS["required"]

    def check_required(validator, required, instance, schema):
        checking_type = schema.get('type', None)
        if checking_type in definitions.keys():
            for property in required:
                if property not in instance:
                    yield ValidationError(f"{property!r} is a required property")
        else:
            yield from validate_required(validator, required, instance, schema)

    return validators.extend(
        validator_class, {"required": check_required},
    )


def extend_coerce_types(validator_class):
    validate_properties = validator_class.VALIDATORS["properties"]

    def coerce_types(validator, properties, instance, schema):
        for property, subschema in properties.items():
            property_value = instance.get(property, None)
            intended_type = subschema.get('type')
            if property_value is None or intended_type not in parsers.keys():
                # Not present or specified, do nothing
                continue
            elif isinstance(property_value, str) and property_value.startswith('#/'):
                # Relative path within spec, resolve it
                property_value = instance.resolve_key(property_value)

            spec_type = parsers[intended_type]
            if not isinstance(property_value, spec_type):
                # If not already of the correct spec type
                # try to coerce it using the class constructor
                # This will cast generic NestedDict to SpecBase's
                # or single values if the constructor accepts that
                property_value = spec_type.construct(property_value,
                                                     parent=instance)
            instance[property] = property_value

        yield from validate_properties(
            validator, properties, instance, schema,
        )

    return validators.extend(
        validator_class, {"properties": coerce_types},
    )


if __name__ == '__main__':
    nest = SpecTree.from_string(toml_def)

    type_checker = validators.Draft202012Validator.TYPE_CHECKER.redefine_many(
        definitions=definitions
    )

    Validator = validators.extend(
        validators.Draft202012Validator,
        type_checker=type_checker,
    )
    Validator = extend_check_required(Validator)
    Validator = extend_coerce_types(Validator)

    validator = Validator(schema=tvips_schema)
    print(validator.is_valid(nest['my_tvips_dataset']))
