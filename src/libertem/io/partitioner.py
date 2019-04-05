import itertools

from libertem.common import Slice, Shape


class Partitioner3D(object):
    def __init__(self):
        pass

    def get_slices(self, shape, num_partitions):
        """
        partition a 3D dataset ("list of frames") along the first axis,
        yielding the partition slice, and additionally start and stop frame
        indices for each partition.
        """
        num_frames = shape.nav.size
        f_per_part = num_frames // num_partitions

        c0 = itertools.count(start=0, step=f_per_part)
        c1 = itertools.count(start=f_per_part, step=f_per_part)
        for (start, stop) in zip(c0, c1):
            if start >= num_frames:
                break
            stop = min(stop, num_frames)
            part_slice = Slice(
                origin=(
                    start, 0, 0,
                ),
                shape=Shape(((stop - start),) + tuple(shape.sig),
                            sig_dims=shape.sig.dims)
            )
            yield part_slice, start, stop
