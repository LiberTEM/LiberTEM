import math
from typing import Any, Optional, overload
from collections.abc import Generator, Sequence

import numpy as np

from libertem.common.math import prod, count_nonzero
from libertem.common.shape import Shape, ShapeLike


class SliceUsageError(ValueError):
    """
    Raised when a Slice is incorrectly instantiated or used
    """


class Slice:
    """
    A n-dimensional slice, defined by origin and shape

    Parameters
    ----------
    origin : tuple of int
        global "top-left" coordinates of this slice
    shape : Shape instance
        the size of this slice
    """

    __slots__ = ["origin", "shape"]

    def __init__(self, origin: Sequence[int], shape: Shape):
        self.origin = tuple(origin)
        self.shape = shape
        if len(self.origin) != len(self.shape):
            raise SliceUsageError(
                ("cannot build slice with dimensionality of shape/origin mismatch (%d vs %d); "
                 "origin=%r, shape=%r") % (
                     len(self.origin), len(self.shape), self.origin, self.shape,
                 )
            )
        if not isinstance(shape, Shape):
            raise SliceUsageError("please use libertem.common.Shape instance as shape parameter")

    def __repr__(self) -> str:
        return f"<Slice origin={self.origin!r} shape={self.shape!r}>"

    def __hash__(self) -> int:
        # enables using a Slice as a key in dict, an item in sets etc.
        # in this case important for use as cache key for our mask container
        return hash((self.origin, tuple(self.shape)))

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Slice) and (
            self.shape == other.shape and self.origin == other.origin
        )

    @classmethod
    def from_shape(cls, shape: Sequence[int], sig_dims: int) -> "Slice":
        """
        Construct a `Slice` at zero-origin from `shape` and `sig_dims`.
        """
        return Slice(
            origin=(0,) * len(shape),
            shape=Shape(shape, sig_dims=sig_dims),
        )

    def intersection_with(self, other: "Slice") -> "Slice":
        """
        Calculate the intersection between this slice and `other`. May result in
        dimensions that are zero, which means that there is no intersection.

        Returns
        -------
        slice : Slice
            the intersection between this and the other slice
        """
        if len(self.origin) != len(other.origin):
            raise SliceUsageError(
                ("cannot intersect slices with different dimensionality (%s vs %s); "
                 "self.origin=%r, other.origin=%r") % (
                    len(self.origin), len(other.origin), self.origin, other.origin,
                )
            )
        if self.shape.sig.dims != other.shape.sig.dims:
            raise SliceUsageError(
                "cannot intersect slices with different signal dimensionality ({} vs {})".format(
                    self.shape.sig.dims, other.shape.sig.dims
                )
            )
        new_origin = tuple(
            max(o1, o2)
            for (o1, o2) in zip(self.origin, other.origin)
        )
        new_shape = [
            min(
                (o1 + s1) - no,
                (o2 + s2) - no,
            )
            for (o1, o2, no, s1, s2) in zip(
                    self.origin, other.origin, new_origin, self.shape, other.shape
            )
        ]
        new_shape = [max(0, s) for s in new_shape]
        result = Slice(
            origin=new_origin,
            shape=Shape(new_shape, sig_dims=self.shape.sig.dims),
        )
        return result

    def is_null(self) -> bool:
        """
        If any part of our shape is zero, this slice doesn't span any data and is null / empty.
        """
        return any(s == 0 for s in self.shape)

    def shift(self, other: "Slice") -> "Slice":
        """
        make a new ``Slice`` with origin relative to ``other.origin``
        and the same shape as this ``Slice``

        useful for translating to the local coordinate system of ``other``
        """
        if len(self.origin) != len(other.origin):
            raise SliceUsageError(
                "cannot shift slices with different "
                f"dimensionality ({self.origin} vs {other.origin})"
            )
        return Slice(origin=tuple(our_coord - their_coord
                                  for (our_coord, their_coord) in zip(self.origin, other.origin)),
                     shape=self.shape)

    def shift_by(self, offset: Sequence[int]) -> "Slice":
        """
        Return a new slice with the origin moved by the supplied offset
        and the same shape
        """
        if len(self.origin) != len(offset):
            raise SliceUsageError(
                "cannot shift slices with different "
                f"dimensionality ({self.origin} vs {offset})"
            )
        return Slice(
            origin=tuple(
                our_coord + off
                for (our_coord, off)
                in zip(self.origin, offset)
            ),
            shape=self.shape,
        )

    @overload
    def get(
        self,
        arr: None = None,
        sig_only: bool = False,
        nav_only: bool = False
    ) -> tuple[slice, ...]: ...

    @overload
    def get(
        self,
        arr: np.ndarray,
        sig_only: bool = False,
        nav_only: bool = False
    ) -> np.ndarray: ...

    def get(
        self,
        arr: Optional[np.ndarray] = None,
        sig_only: bool = False,
        nav_only: bool = False
    ):
        """
        Get a standard python tuple-of-slice-object which can be used
        to slice any compatible numpy.ndarray

        Parameters
        ----------
        arr
            something implementing the slice interface. if given, returns arr[slice]
        sig_only : bool
            get a signal-only slice for frames/masks
        nav_only : bool
            get a nav-only slice, for example for indexing something that is shaped like
            the navigation dimensions of this Slice.

        Returns
        -------
        tuple of slice objects
            returns standard python slices computed from
            our origin+shape model or arr indexed with this slicing
            if arr is given


        Examples
        --------
        >>> import numpy as np
        >>> from libertem.common import Slice, Shape
        >>> s = Slice(shape=Shape((16, 16, 4, 4), sig_dims=2), origin=(0, 0, 12, 12))
        >>> data = np.ones((16, 16))
        >>> data[s.get(sig_only=True)]
        array([[1., 1., 1., 1.],
               [1., 1., 1., 1.],
               [1., 1., 1., 1.],
               [1., 1., 1., 1.]])
        """
        if sig_only:
            o, s = self.origin, self.shape
            slice_ = tuple(
                slice(o[i], (o[i] + s[i]))
                for i in range(s.nav.dims, s.sig.dims + s.nav.dims)
            )
        elif nav_only:
            o, s = self.origin, self.shape
            slice_ = tuple(
                slice(o[i], (o[i] + s[i]))
                for i in range(s.nav.dims)
            )
        else:
            slice_ = self._get()
        if arr is not None:
            if sig_only:
                # Skip the supposed nav dimensions of the data
                return arr[(Ellipsis, ) + slice_]
            else:
                # for nav_only, we return the full remaining dimensions anyway
                # if arr has more dimensions than the slice
                return arr[slice_]
        else:
            return slice_

    def _get(self):
        """
        Direct conversion from Slice to tuple(slice, ...) without options
        """
        return tuple(slice(o, (o + s)) for (o, s) in zip(self.origin, self.shape))

    def discard_nav(self) -> "Slice":
        """
        returns a copy with the origin/shape zeroed in the nav dimensions

        this is used to create uniform cache keys
        """
        o, s, sig_dims = self._discard_nav_key()
        new_shape = Shape(s, sig_dims=sig_dims)
        return Slice(origin=o, shape=new_shape)

    def _discard_nav_key(self) -> tuple[tuple[int, ...], tuple[int, ...], int]:
        """
        Construct a hashable tuple of the Slice with a zero-length nav dimensions

        Functions as discard_nav but avoids Shape and Slice constructor overheads
        """
        o, s = self.origin, self.shape
        nav_dims = s.nav_dims
        zero_nav = (0,) * nav_dims
        return (zero_nav + o[nav_dims:], zero_nav + s._sig_shape, s.sig_dims)

    def subslices(self, shape: ShapeLike) -> Generator["Slice", None, None]:
        """
        Generator for all subslices of this slice with dimensions
        specified by ``shape``.

        Parameters
        ----------
        shape : tuple of int or Shape
            the shape of each sub-slice

        Yields
        ------
        Slice
            all subslices, in fast-access order
        """
        # example: self.shape=(3, 1, 1, 1), subslice shape=(2, 1, 1, 1)
        # math.ceil(3/2) = math.ceil(1.5) = 2 -> we need two subslices across the y dimension
        shape = Shape(shape, sig_dims=self.shape.sig.dims)
        if self.shape.dims != shape.dims:
            raise SliceUsageError(
                ("cannot create subslices with different dimensionality (%d vs %d); "
                 "self.shape=%r, shape=%r") % (
                    self.shape.dims, shape.dims, self.shape, shape,
                )
            )
        ni = tuple(math.ceil(s1 / s)
                   for (s1, s) in zip(self.shape, shape))

        def _make_slice(origin: tuple[int, ...], new_shape: Shape) -> Slice:
            sig_dims = new_shape.sig.dims
            # this makes sure that the border tiles have the correct shape set
            new_shape_tuple = tuple(
                min(ns, so + s - o)
                for (ns, so, s, o) in zip(new_shape, self.origin, self.shape, origin)
            )
            new_shape = Shape(new_shape_tuple, sig_dims=sig_dims)
            for x in new_shape_tuple:
                assert x > 0, \
                    "invalid shape: {!r} while subslicing {!r} with {!r} (origin={!r})".format(
                        new_shape, self.shape, shape, origin
                    )
            return Slice(
                origin=origin,
                shape=new_shape,
            )

        return (
            _make_slice(origin=tuple(
                o + i * s
                for (o, i, s) in zip(self.origin, indexes, shape)
            ), new_shape=Shape(tuple(shape), sig_dims=self.shape.sig.dims))

            for indexes in np.ndindex(ni)
        )

    @property
    def nav(self) -> "Slice":
        """
        Returns a new Slice, with sig_dims=0, limited to the nav part
        """
        return Slice(
            origin=self.origin[:self.shape.nav.dims],
            shape=self.shape.nav,
        )

    @property
    def sig(self) -> "Slice":
        """
        Returns a new Slice, limited to the sig part
        """
        return Slice(
            origin=self.origin[self.shape.nav.dims:],
            shape=self.shape.sig,
        )

    def flatten_nav(self, containing_shape: ShapeLike) -> "Slice":
        sig_dims = self.shape.sig.dims
        nav_dims = self.shape.dims - sig_dims
        containing_shape = tuple(containing_shape)[:nav_dims]
        origin = self.origin[:nav_dims]

        # validation for the nav_shape:
        # what are the preconditions that allow flattening?
        #
        # - nav part of the shape: must be in the form of:
        #
        #   (1,  1,  ...,  N, M, M, ...)
        #
        #   where N<=M and M is the corresponding part of
        #   the shape of the dataset.
        #
        # - the origin must match the shape in the following way:
        #
        #   (o1, o2, ..., oi, 0, 0, ...)
        #
        #   where all oj are arbitraty (but in bounds)
        #
        state = 0
        for cs, s, o in zip(containing_shape, self.shape.nav, origin):
            if state == 0:
                if s != 1:
                    state = 1
                    assert s <= cs, "invalid nav_shape #1"
            elif state == 1:
                assert s == cs, "invalid nav_shape #2"
                assert o == 0, "invalid origin"

        nav_origin = np.ravel_multi_index(
            origin,
            containing_shape
        )
        nav_shape = prod(self.shape.nav)
        return Slice(
            origin=(nav_origin,) + self.origin[nav_dims:],
            shape=Shape((nav_shape,) + tuple(self.shape.sig), sig_dims=sig_dims)
        )

    def adjust_for_roi(self, roi: Optional[np.ndarray]) -> "Slice":
        """
        Make a new slice that has origin and shape modified according to `roi`.
        """
        if roi is None:
            return self
        roi = roi.reshape(-1)
        assert self.shape.nav.dims == 1
        s_o = self.origin[0]
        s_s = self.shape[0]
        # We need to find how many 1s there are for all previous partitions, to know
        # the origin; then we count how many 1s there are in our partition
        # to find our shape.
        origin = count_nonzero(roi[:s_o])
        shape = count_nonzero(roi[s_o:s_o + s_s])
        sig_dims = self.shape.sig.dims
        return Slice(
            origin=(origin,) + self.origin[-sig_dims:],
            shape=Shape((shape,) + tuple(self.shape.sig), sig_dims=sig_dims),
        )

    def clip_to(self, shape: Shape):
        other_slice = Slice((0,) * shape.dims, shape)
        return self.intersection_with(other_slice)

    def __getstate__(self) -> dict[str, Any]:
        return {
            k: getattr(self, k)
            for k in self.__slots__
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        for k, v in state.items():
            setattr(self, k, v)
