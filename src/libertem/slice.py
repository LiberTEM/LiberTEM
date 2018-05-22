import multiprocessing
import numpy as np


class Slice(object):
    def __init__(self, origin, shape):
        """
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

    def shift(self, other):
        """
        make a new ``Slice`` with origin relative to ``other.origin``
        and the same shape as this ``Slice``
        """
        assert len(other.origin) == len(self.origin)
        return Slice(origin=tuple(their_coord - our_coord
                                  for (our_coord, their_coord) in zip(self.origin, other.origin)),
                     shape=self.shape)

    def get(self, arr=None):
        o, s = self.origin, self.shape
        if arr:
            return arr[
                o[0]:(o[0] + s[0]),
                o[1]:(o[1] + s[1]),
                o[2]:(o[2] + s[2]),
                o[3]:(o[3] + s[3]),
            ]
        else:
            return (
                slice(o[0], (o[0] + s[0])),
                slice(o[1], (o[1] + s[1])),
                slice(o[2], (o[2] + s[2])),
                slice(o[3], (o[3] + s[3])),
            )

    def subslices(self, shape):
        """
        Parameters
        ----------
        shape : (int, int, int, int)
            the shape of each sub-slice

        Yields
        ------
        Slice
            all subslices, in fast-access order
        """
        for i in range(len(self.shape)):
            assert self.shape[i] % shape[i] == 0
        ny = self.shape[0] // shape[0]
        nx = self.shape[1] // shape[1]
        nv = self.shape[2] // shape[2]
        nu = self.shape[3] // shape[3]

        return (
            Slice(
                origin=(
                    self.origin[0] + y * shape[0],
                    self.origin[1] + x * shape[1],
                    self.origin[2] + v * shape[2],
                    self.origin[3] + u * shape[3],
                ),
                shape=shape,
            )
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
        # FIXME: allow for partitions samller than one scan row
        # FIXME: allow specifying the "aspect ratio" for a partition?
        num_frames = datashape[0] * datashape[1]
        bytes_per_frame = framesize * np.typeDict[str(dtype)]().itemsize
        frames_per_partition = target_size // bytes_per_frame
        num_partitions = num_frames // frames_per_partition
        num_partitions = max(min_num_partitions, num_partitions)

        # number of partitions should evenly divide number of scan rows:
        assert datashape[1] % num_partitions == 0,\
            "%d %% %d != 0" % (datashape[1], num_partitions)

        return (datashape[0] // num_partitions, datashape[1], datashape[2], datashape[3])
