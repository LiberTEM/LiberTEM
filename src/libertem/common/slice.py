import math
import numpy as np
from libertem.common.shape import Shape


class Slice(object):
    __slots__ = ["origin", "shape"]

    def __init__(self, origin, shape):
        """
        A n-dimensional slice, defined by origin and shape

        Parameters
        ----------
        origin : tuple of int
            global "top-left" coordinates of this slice
        shape : Shape instance
            the size of this slice
        """
        self.origin = tuple(origin)
        self.shape = shape
        if len(self.origin) != len(self.shape):
            raise ValueError(
                "cannot build slice with dimensionality of shape/origin mismatch (%d vs %d)" % (
                    len(self.origin), len(self.shape)
                )
            )
        if not hasattr(shape, 'sig'):
            raise ValueError("please use libertem.common.Shape instance as shape parameter")

    def __repr__(self):
        return "<Slice origin=%r shape=%r>" % (self.origin, self.shape)

    def __hash__(self):
        # enables using a Slice as a key in dict, an item in sets etc.
        # in this case important for use as cache key for our mask container
        return hash((self.origin, tuple(self.shape)))

    def __eq__(self, other):
        return self.shape == other.shape and self.origin == other.origin

    def intersection_with(self, other):
        """
        Calculate the intersection between this slice and `other`. May result in
        dimensions that are zero, which means that there is no intersection.

        Returns
        -------
        slice : Slice
            the intersection between this and the other slice
        """
        if len(self.origin) != len(other.origin):
            raise ValueError("cannot intersect slices with different dimensionality")
        if len(self.shape.sig) != len(other.shape.sig):
            raise ValueError("cannot intersect slices with different signal dimensionality")
        new_origin = tuple([
            max(o1, o2)
            for (o1, o2) in zip(self.origin, other.origin)
        ])
        new_shape = tuple([
            min(
                (o1 + s1) - no,
                (o2 + s2) - no,
            )
            for (o1, o2, no, s1, s2) in zip(
                    self.origin, other.origin, new_origin, self.shape, other.shape
            )
        ])
        new_shape = [max(0, s) for s in new_shape]
        result = Slice(
            origin=new_origin,
            shape=Shape(new_shape, sig_dims=len(self.shape.sig)),
        )
        return result

    def is_null(self):
        """
        If any part of our shape is zero, this slice doesn't span any data and is null / empty.
        """
        return any(s == 0 for s in self.shape)

    def shift(self, other):
        """
        make a new ``Slice`` with origin relative to ``other.origin``
        and the same shape as this ``Slice``

        useful for translating to the local coordinate system of ``other``
        """
        if len(self.origin) != len(other.origin):
            raise ValueError("cannot shift slices with different dimensionality")
        return Slice(origin=tuple(our_coord - their_coord
                                  for (our_coord, their_coord) in zip(self.origin, other.origin)),
                     shape=self.shape)

    def get(self, arr=None, sig_only=False, nav_only=False):
        """
        Get a standard python tuple-of-slice-object which can be used
        to slice any compatible ndarray

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
        >>> s = Slice(shape=Shape((16, 16, 16, 16), sig_dims=2), origin=(0, 0, 4, 4))
        >>> data = np.ones((16, 16))
        >>> data[s.get(sig_only=True)
        ...
        """
        if sig_only:
            o, s = self.origin, self.shape
            slice_ = tuple([
                slice(o[i], (o[i] + s[i]))
                for i in range(s.nav.dims, s.sig.dims + s.nav.dims)
            ])
        elif nav_only:
            o, s = self.origin, self.shape
            slice_ = tuple([
                slice(o[i], (o[i] + s[i]))
                for i in range(s.nav.dims)
            ])
        else:
            slice_ = tuple([
                slice(o, (o + s))
                for (o, s) in zip(self.origin, self.shape)
            ])
        if arr is not None:
            return arr[slice_]
        else:
            return slice_

    def discard_nav(self):
        """
        returns a copy with the nav dimensions zeroed

        this is used to create uniform cache keys
        """
        o, s = self.origin, self.shape
        return Slice(origin=tuple([0] * s.nav.dims) + o[s.nav.dims:], shape=s)

    def subslices(self, shape):
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
        if len(self.shape) != len(shape):
            raise ValueError("cannot create subslices with different dimensionality (%d vs %d)" % (
                len(self.shape), len(shape)
            ))
        ni = tuple([math.ceil(s1 / s)
                    for (s1, s) in zip(self.shape, shape)])

        def _make_slice(origin, new_shape):
            sig_dims = len(new_shape.sig)
            # this makes sure that the border tiles have the correct shape set
            new_shape = tuple([
                min(ns, so + s - o)
                for (ns, so, s, o) in zip(new_shape, self.origin, self.shape, origin)
            ])
            new_shape = Shape(new_shape, sig_dims=sig_dims)
            for x in new_shape:
                assert x > 0, "invalid shape: %r while subslicing %r with %r (origin=%r)" % (
                    new_shape, self.shape, shape, origin
                )
            return Slice(
                origin=origin,
                shape=new_shape,
            )

        return (
            _make_slice(origin=tuple([
                o + i * s
                for (o, i, s) in zip(self.origin, indexes, shape)
            ]), new_shape=Shape(tuple(shape), sig_dims=len(self.shape.sig)))

            for indexes in np.ndindex(ni)
        )

    def subslice_from_offset(self, offset, length):
        """
        in scan dimensions
        """
        raise Exception("nope")
