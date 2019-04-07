import numpy as np

from libertem.job.base import Task


def check_cast(fromvar, tovar):
    if not np.can_cast(fromvar.dtype, tovar.dtype, casting='safe'):
        # FIXME exception or warning?
        raise TypeError("Unsafe automatic casting from %s to %s" % (fromvar.dtype, tovar.dtype))


def merge_assign(dest, src):
    for k in dest:
        check_cast(dest[k], src[k])
        dest[k][:] = src[k]


class UDFTask(Task):
    def __init__(self, partition, idx, make_buffers, init, fn):
        super().__init__(partition=partition, idx=idx)
        self._make_buffers = make_buffers
        self._init = init
        self._fn = fn

    def __call__(self):
        result_buffers = self._make_buffers()
        for buf in result_buffers.values():
            buf.set_shape_partition(self.partition)
            buf.allocate()
        if self._init is not None:
            kwargs = self._init(self.partition)
        else:
            kwargs = {}
        kwargs.update(result_buffers)
        for tile in self.partition.get_tiles(full_frames=True):
            for frame_idx, frame in enumerate(tile.data):
                buffer_views = {}
                for k, buf in result_buffers.items():
                    buffer_views[k] = buf.get_view_for_frame(
                        partition=self.partition,
                        tile=tile,
                        frame_idx=frame_idx
                    )
                kwargs.update(buffer_views)
                self._fn(frame=frame, **kwargs)
        return result_buffers, self.partition


def make_udf_tasks(dataset, fn, init, make_buffers):
    return (
        UDFTask(partition=partition, idx=idx, fn=fn, init=init, make_buffers=make_buffers)
        for idx, partition in enumerate(dataset.get_partitions())
    )
