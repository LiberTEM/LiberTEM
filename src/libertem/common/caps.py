class Capability(object):
    """
    represents a single capability
    """
    def __init__(self, label):
        self.label = label

    def __str__(self):
        return self.label

    def __repr__(self):
        return "<Capability: %s>" % self.label


class BaseCaps(object):
    """
    represents capabilities for a class or function
    """

    def __init__(self, caps):
        for cap in caps:
            if cap not in self.ALL_CAPS:
                raise ValueError("unknown capability: %r" % cap)
        self._caps = set(caps)

    def __call__(self, fn, *args, **kwargs):
        setattr(fn, self._ATTR_NAME, self._caps)

        def _inner(*args, **kwargs):
            return fn(*args, **kwargs)

        return fn

    @classmethod
    def has_cap(cls, obj, cap):
        if cap not in cls.ALL_CAPS:
            raise ValueError("unknown capability: %r" % cap)
        return cap in cls.get_caps(obj)

    @classmethod
    def get_caps(cls, obj):
        return getattr(obj, cls._ATTR_NAME, set())


def make_caps(*args, **kwargs):
    name = kwargs.pop('name')
    caps = {
        arg: Capability(arg)
        for arg in args
    }
    attrs = {
        'ALL_CAPS': set(caps.values()),
        '_ATTR_NAME': "_%s" % name.upper(),
    }
    attrs.update(caps)
    return type(name, (BaseCaps,), attrs)
