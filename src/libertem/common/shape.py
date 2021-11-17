import operator
import functools
from typing import Any, Dict, Iterator, Sequence, Tuple, Union, overload


class Shape:
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

    __slots__ = ["_sig_dims", "_nav_shape", "_sig_shape"]

    def __init__(self, shape: "ShapeLike", sig_dims: int):
        nav_dims = len(shape) - sig_dims
        self._sig_dims = sig_dims
        self._nav_shape = tuple(shape)[:nav_dims]
        self._sig_shape = tuple(shape)[nav_dims:]

    @property
    def nav(self) -> "Shape":
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
    def sig(self) -> "Shape":
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

    def to_tuple(self) -> Tuple[int, ...]:
        return tuple(self)

    @property
    def size(self) -> int:
        """
        Number of elements covered by this shape

        Examples
        --------

        >>> from libertem.common import Shape
        >>> s = Shape((16, 16), sig_dims=2)
        >>> s.size
        256
        """
        shape_tuple = tuple(self)
        if len(shape_tuple) == 0:
            return 0
        return functools.reduce(operator.mul, shape_tuple)

    def flatten_nav(self) -> "Shape":
        """
        Returns a new Shape that is flat in the navigation dimensions

        Examples
        --------

        >>> from libertem.common import Shape
        >>> s = Shape((5, 5, 16, 16), sig_dims=2)
        >>> s.flatten_nav()
        (25, 16, 16)
        """
        return Shape(shape=(self.nav.size,) + self._sig_shape, sig_dims=self._sig_dims)

    def flatten_sig(self) -> "Shape":
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
    def dims(self) -> int:
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

    def __iter__(self) -> Iterator[int]:
        """
        Iterate over all parts of the shape
        """
        return iter(self._nav_shape + self._sig_shape)

    def __repr__(self) -> str:
        return repr(tuple(self))

    @overload
    def __getitem__(self, k: int) -> int: ...

    @overload
    def __getitem__(self, k: slice) -> Tuple[int, ...]: ...

    def __getitem__(self, k):
        return tuple(self)[k]

    def __len__(self) -> int:
        return len(self._sig_shape) + len(self._nav_shape)

    def __eq__(self, other: object) -> bool:
        """
        Shape instances are equal if both the shape tuple and the number of signal dimensions
        are equal.
        """
        if not isinstance(other, Shape):
            raise NotImplementedError()
        dims_eq = self._sig_dims == other._sig_dims
        values_eq = tuple(self) == tuple(other)
        return dims_eq and values_eq

    def __add__(self, other: object) -> "Shape":
        """
        Right addition of a Shape object and a tuple.
        Right addition adds the tuple to the signal dimensions of the Shape object.
        """
        if isinstance(other, tuple):
            return Shape(self._nav_shape + self._sig_shape + other,
                 sig_dims=self.sig.dims + len(other))
        else:
            raise NotImplementedError()

    def __radd__(self, other: object) -> "Shape":
        """
        Left addition of a Shape object and a tuple
        Left addition adds the tuple to the navigation dimensions of the Shape object.
        """
        if isinstance(other, tuple):
            return Shape(self._nav_shape + other + self._sig_shape,
                 sig_dims=self.sig.dims)
        else:
            return NotImplemented

    def __getstate__(self) -> Dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in self.__slots__
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        for k, v in state.items():
            setattr(self, k, v)


ShapeLike = Union[Shape, Sequence[int]]
