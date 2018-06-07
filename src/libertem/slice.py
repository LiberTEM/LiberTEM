import math
import multiprocessing
import numpy as np


class Slice(object):
    def __init__(self, origin, shape):
        """
        A slice into a 4D dataset, defined by origin and shape

        Parameters
        ----------
        origin : (int, int) or (int, int, int, int)
            global top-left coordinates of this slice, will be "broadcast" to 4D
        shape : (int, int, int, int)
            the size of this slice
        """
        if len(origin) == 2:
            origin = (origin[0], origin[1], 0, 0)
        self.origin = tuple(origin)
        self.shape = tuple(shape)
        # TODO: allow to use Slice objects directly for... slices!
        # arr[slice]
        # or use a Slicer object, a little bit like hyperspy .isig, .inav?
        # Slicer(arr)[slice]
        # can we implement some kind of slicer interface? __slice__?

    def __repr__(self):
        return "<Slice origin=%r shape=%r>" % (self.origin, self.shape)

    def __hash__(self):
        return hash((self.origin, self.shape))

    def __eq__(self, other):
        return self.shape == other.shape and self.origin == other.origin

    def shift(self, other):
        """
        make a new ``Slice`` with origin relative to ``other.origin``
        and the same shape as this ``Slice``
        """
        assert len(other.origin) == len(self.origin)
        return Slice(origin=tuple(their_coord - our_coord
                                  for (our_coord, their_coord) in zip(self.origin, other.origin)),
                     shape=self.shape)

    def get(self, arr=None, signal_only=False):
        """
        Get a standard python tuple-of-slice-object which can be used
        to slice any 2D/4D ndarray

        Parameters
        ----------
        arr : sliceable or None
            if given, returns arr[slice]
        signal_only : bool (default False)
            get a 2D slice for frames/masks

        Returns
        -------
        (slice, slice, slice, slice) or (slice, slice)
            returns standard python slices computed from
            our origin+shape model or arr indexed with this slicing
            if arr is given
        """
        o, s = self.origin, self.shape
        if signal_only:
            slice_ = (
                slice(o[2], (o[2] + s[2])),
                slice(o[3], (o[3] + s[3])),
            )
        else:
            slice_ = (
                slice(o[0], (o[0] + s[0])),
                slice(o[1], (o[1] + s[1])),
                slice(o[2], (o[2] + s[2])),
                slice(o[3], (o[3] + s[3])),
            )
        if arr is not None:
            return arr[slice_]
        else:
            return slice_

    def discard_scan(self):
        """
        returns a copy with the scan dimensions zeroed
        """
        o, s = self.origin, self.shape
        return Slice(origin=(0, 0) + o[2:], shape=s)

    def subslices(self, shape):
        """
        Generator for all subslices of this slice with dimensions
        specified by ``shape``.

        Parameters
        ----------
        shape : (int, int, int, int)
            the shape of each sub-slice

        Yields
        ------
        Slice
            all subslices, in fast-access order
        """
        # TODO: maybe find a more general formulation for n dimensions

        # example: self.shape=(3, 1, 1, 1), subslice shape=(2, 1, 1, 1)
        # math.ceil(3/2) = math.ceil(1.5) = 2 -> we need two subslices across the y dimension
        ny = math.ceil(self.shape[0] / shape[0])
        nx = math.ceil(self.shape[1] / shape[1])
        nv = math.ceil(self.shape[2] / shape[2])
        nu = math.ceil(self.shape[3] / shape[3])

        def _make_slice(origin, new_shape):
            # this makes sure that the border tiles have the correct shape set
            new_shape = (
                min(new_shape[0], self.origin[0] + self.shape[0] - origin[0]),
                min(new_shape[1], self.origin[1] + self.shape[1] - origin[1]),
                min(new_shape[2], self.origin[2] + self.shape[2] - origin[2]),
                min(new_shape[3], self.origin[3] + self.shape[3] - origin[3]),
            )
            for x in new_shape:
                assert x > 0, "invalid shape: %r while subslicing %r with %r (origin=%r)" % (
                    new_shape, self.shape, shape, origin
                )
            return Slice(
                origin=origin,
                shape=new_shape,
            )

        return (
            _make_slice(origin=(
                self.origin[0] + y * shape[0],
                self.origin[1] + x * shape[1],
                self.origin[2] + v * shape[2],
                self.origin[3] + u * shape[3],
            ), new_shape=shape)
            for y in range(ny)
            for x in range(nx)
            for v in range(nv)
            for u in range(nu)
        )

    @classmethod
    def partition_shape(cls, datashape, framesize, dtype, target_size, min_num_partitions=None):
        """
        Calculate partition shape for the given ``target_size``

        Returns
        -------
        (int, int, int, int)
            the shape calculated from the given parameters
        """
        min_num_partitions = min_num_partitions or 2 * multiprocessing.cpu_count()
        # FIXME: allow for partitions smaller than one scan row
        # FIXME: allow specifying the "aspect ratio" for a partition?
        num_frames = datashape[0] * datashape[1]
        bytes_per_frame = framesize * np.typeDict[str(dtype)]().itemsize
        frames_per_partition = target_size // bytes_per_frame
        num_partitions = num_frames // frames_per_partition
        num_partitions = max(min_num_partitions, num_partitions)

        # number of partitions should evenly divide number of scan rows:
        # assert datashape[1] % num_partitions == 0,\
        #     "%d %% %d != 0 (datashape=%r)" % (datashape[1], num_partitions, datashape)

        return (max(1, datashape[0] // num_partitions), datashape[1], datashape[2], datashape[3])
