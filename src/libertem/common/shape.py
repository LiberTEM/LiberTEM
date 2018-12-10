import operator
import functools


class Shape(object):
    __slots__ = ["_sig_dims", "_nav_shape", "_sig_shape"]

    def __init__(self, shape, sig_dims):
        """
        Create a Shape that knows how many dimensions are part of navigation/signal.
        It is assumed that the signal is in the last `sig_dims` dimensions.

        Parameters
        ----------
        shape : tuple of int
            the shape we want to work with, as n-tuple (like numpy array shapes)
        sig_dims : int
            the number of dimensions that are considered part of the signal
        """
        nav_dims = len(shape) - sig_dims
        self._sig_dims = sig_dims
        self._nav_shape = tuple(shape[:nav_dims])
        self._sig_shape = tuple(shape[nav_dims:])

    @property
    def nav(self):
        """
        Crop to navigation dimensions

        Returns
        -------
        shape : Shape
            like this shape, but without the signal dimensions

        Examples
        --------

        >>> from libertem.common import Shape
        >>> s = Shape((5, 5, 16, 16), sig_dims=2)
        >>> s.nav
        (5, 5)
        """
        return Shape(shape=self._nav_shape, sig_dims=0)

    @property
    def sig(self):
        """
        Crop to signal dimensions

        Returns
        -------
        shape : Shape
            like this shape, but without the navigation dimensions

        Examples
        --------

        >>> from libertem.common import Shape
        >>> s = Shape((5, 5, 16, 16), sig_dims=2)
        >>> s.sig
        (16, 16)
        """
        return Shape(shape=self._sig_shape, sig_dims=self._sig_dims)

    def to_tuple(self):
        return tuple(self)

    @property
    def size(self):
        """
        Nunmber of elements covered by this shape

        Examples
        --------

        >>> from libertem.common import Shape
        >>> s = Shape((16, 16), sig_dims=2)
        >>> s.size
        256
        """
        return functools.reduce(operator.mul, self)

    def flatten_nav(self):
        """
        Flatten in the navigation dimensions

        Examples
        --------

        >>> from libertem.common import Shape
        >>> s = Shape((5, 5, 16, 16), sig_dims=2)
        >>> s.flatten_nav()
        (25, 16, 16)
        """
        return Shape(shape=(self.nav.size,) + self._sig_shape, sig_dims=self._sig_dims)

    def flatten_sig(self):
        """
        Flatten in the signal dimensions

        Examples
        --------

        >>> from libertem.common import Shape
        >>> s = Shape((5, 5, 16, 16), sig_dims=2)
        >>> s.flatten_sig()
        (5, 5, 256)
        """
        return Shape(shape=self._nav_shape + (self.sig.size,), sig_dims=1)

    @property
    def dims(self):
        """
        Number of dimensions

        Examples
        --------

        >>> from libertem.common import Shape
        >>> s = Shape((5, 5, 16, 16), sig_dims=2)
        >>> s.dims
        4
        >>> s.nav.dims  # creates a new temporary Shape and accesses .dims on it
        2
        >>> s.sig.dims
        2
        """
        return len(self)

    def __iter__(self):
        """
        Iterate over all parts of the shape
        """
        return iter(self._nav_shape + self._sig_shape)

    def __repr__(self):
        return repr(tuple(self))

    def __getitem__(self, k):
        return tuple(self)[k]

    def __len__(self):
        return len(self._sig_shape) + len(self._nav_shape)

    def __eq__(self, other):
        """
        Shape instances are equal if both the shape tuple and the number of signal dimensions
        are equal.
        """
        dims_eq = self._sig_dims == other._sig_dims
        values_eq = tuple(self) == tuple(other)
        return dims_eq and values_eq
