import numpy as np


class WrappedType:
    """
    Base class for non-dict specifications
    Matches the SpecBase interface for validate and construct
    so that we can use schema validation to cast values from the config
    to the appropriate python object type and validate them
    """
    @classmethod
    def validate(cls, checker, instance):
        raise NotImplementedError()

    @classmethod
    def construct(cls, arg, parent=None):
        return arg


class DType(WrappedType):
    """
    Allows specifying 'type = dtype' in a JSON Schema
    and have the value in the config properly cast to
    a numpy dtype and validated
    """
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
