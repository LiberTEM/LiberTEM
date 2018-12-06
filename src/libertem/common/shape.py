import operator
import functools


class Shape(object):
    __slots__ = ["_sig_dims", "_nav_shape", "_sig_shape"]

    def __init__(self, shape, sig_dims):
        nav_dims = len(shape) - sig_dims
        self._sig_dims = sig_dims
        self._nav_shape = shape[:nav_dims]
        self._sig_shape = shape[nav_dims:]

    @property
    def nav(self):
        return Shape(shape=self._nav_shape, sig_dims=0)

    @property
    def sig(self):
        return Shape(shape=self._sig_shape, sig_dims=self._sig_dims)

    def to_tuple(self):
        return tuple(self)

    @property
    def size(self):
        return functools.reduce(operator.mul, self)

    def flatten_nav(self):
        return Shape(shape=(self.nav.size,) + self._sig_shape, sig_dims=self._sig_dims)

    def flatten_sig(self):
        return Shape(shape=self._nav_shape + (self.sig.size,), sig_dims=1)

    @property
    def dims(self):
        return len(self)

    def __iter__(self):
        return iter(self._nav_shape + self._sig_shape)

    def __repr__(self):
        return repr(tuple(self))

    def __getitem__(self, k):
        return tuple(self)[k]

    def __len__(self):
        return len(self._sig_shape) + len(self._nav_shape)

    def __eq__(self, other):
        dims_eq = self._sig_dims == other._sig_dims
        values_eq = tuple(self) == tuple(other)
        return dims_eq and values_eq
