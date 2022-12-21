from jsonschema import validators
from jsonschema.exceptions import ValidationError


class MissingValue:
    ...


def extend_check_required(validator_class, types):
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


def extend_coerce_types(validator_class, types):
    validate_properties = validator_class.VALIDATORS["properties"]

    def coerce_types(validator, properties, instance, schema):
        for property, subschema in properties.items():
            default = subschema.get('default', MissingValue())
            property_value = instance.get(property, default)
            intended_type = subschema.get('type')
            if isinstance(property_value, MissingValue):
                # Not present and no default, do nothing
                continue
            elif isinstance(property_value, str) and property_value.startswith('#/'):
                # Relative path within spec, resolve it
                property_value = instance.resolve_key(property_value)

            spec_type = types.get(intended_type, None)
            if spec_type is not None and not isinstance(property_value, spec_type):
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


def get_validator(schema, types):
    type_checker = validators.Draft202012Validator.TYPE_CHECKER.redefine_many(
        definitions={k: v.validate for k, v in types.items()}
    )

    Validator = validators.extend(
        validators.Draft202012Validator,
        type_checker=type_checker,
    )
    Validator = extend_check_required(Validator, types)
    Validator = extend_coerce_types(Validator, types)
    return Validator(schema=schema)
