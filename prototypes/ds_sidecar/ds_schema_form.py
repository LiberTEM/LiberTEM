import pathlib
import numpy as np
from jsonschema import validators
from jsonschema.exceptions import ValidationError

from file_spec import SpecTree, parsers


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


if __name__ == '__main__':
    nest = SpecTree.from_string(toml_def)

    type_checker = validators.Draft202012Validator.TYPE_CHECKER.redefine_many(
        definitions={k: v.validate for k, v in types.items()}
    )

    Validator = validators.extend(
        validators.Draft202012Validator,
        type_checker=type_checker,
    )
    Validator = extend_check_required(Validator)
    Validator = extend_coerce_types(Validator)

    validator = Validator(schema=tvips_schema)
    validator.validate(nest['my_tvips_dataset'])
