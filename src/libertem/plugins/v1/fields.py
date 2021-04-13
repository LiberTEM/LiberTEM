import jsonschema
import traitlets


mask_schema = {
    "anyOf": [
        {
            "type": "object",
            "properties": {
                "shape": {"const": "disk"},
                "cy": {"type": "number"},
                "cx": {"type": "number"},
                "r": {"type": "number"},
            },
            "required": ["cx", "cy", "r"],
        },
        {
            "type": "object",
            "properties": {
                "shape": {"const": "ring"},
                "cy": {"type": "number"},
                "cx": {"type": "number"},
                "ri": {"type": "number"},
                "ro": {"type": "number"},
            },
            "required": ["cx", "cy", "ri", "ro"],
        },
        {
            "type": "object",
            "properties": {
                "shape": {"const": "all"},
            },
        },
        {
            "type": "object",
            "properties": {
                "shape": {"const": "rect"},
                "x": {"type": "number"},
                "y": {"type": "number"},
                "width": {"type": "number"},
                "height": {"type": "number"},
            },
            "required": ["x", "y", "width", "height"],
        },
    ]
}


class MaskField(traitlets.Dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag(widget_type='mask')

    def validate(self, obj, value):
        try:
            jsonschema.validate(value, mask_schema)
        except jsonschema.ValidationError as e:
            raise traitlets.TraitError(e)
        return value


def make_field_type(name, base_type, tags):
    def _new_init(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tag(**tags)

    return type(name, (base_type,), {'__init__': _new_init})
