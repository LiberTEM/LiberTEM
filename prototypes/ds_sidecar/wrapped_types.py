import numpy as np


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
