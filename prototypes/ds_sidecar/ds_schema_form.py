import pathlib
import numpy as np
from jsonschema import validators
from jsonschema.exceptions import ValidationError

from file_spec import SpecTree, types


toml_def = """
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

[my_dark_frame]
type = 'nparray'
data = [
   [5.0, 6.0, 7.0, 8.0],
   [1.0, 2.0, 3.0, 4.0],
   [5.0, 6.0, 7.0, 8.0],
   [1.0, 2.0, 3.0, 4.0],
]
shape = [2, 8]

[my_roi]
type = 'nparray'
read_as = 'file'
path = './test_roi.npy'
"""

tvips_schema = {
    "type": "dataset",
    "title": "TVIPS dataset",
    "properties": {
        "meta": {
            "type": "file",
            "properties": {
                "dtype": {
                    "type": "dtype",
                    "default": "float32"
                },
            },
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
            "type": "dtype",
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


def extend_check_required(validator_class):
    validate_required = validator_class.VALIDATORS["required"]

    def check_required(validator, required, instance, schema):
        checking_type = schema.get('type', None)
        if checking_type in types.keys():
            for property in required:
                if not isinstance(instance, dict):
                    yield ValidationError('Cannot use "required" in non-dict '
                                          f'specification {checking_type}')
                    continue
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
            default = subschema.get('default', None)
            property_value = instance.get(property, default)
            intended_type = subschema.get('type')
            if property_value is None or intended_type not in types.keys():
                # Not present or specified, do nothing
                continue
            elif isinstance(property_value, str) and property_value.startswith('#/'):
                # Relative path within spec, resolve it
                property_value = instance.resolve_key(property_value)

            spec_type = types[intended_type]
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


def get_validator(schema):
    type_checker = validators.Draft202012Validator.TYPE_CHECKER.redefine_many(
        definitions={k: v.validate for k, v in types.items()}
    )

    Validator = validators.extend(
        validators.Draft202012Validator,
        type_checker=type_checker,
    )
    Validator = extend_check_required(Validator)
    Validator = extend_coerce_types(Validator)
    return Validator(schema=schema)


if __name__ == '__main__':
    nest = SpecTree.from_string(toml_def)
    validator = get_validator(tvips_schema)
    validator.validate(nest['my_tvips_dataset'])
