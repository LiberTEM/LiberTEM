import functools
import numpy as np


class ResultBuffer(object):
    def __init__(self, kind, extra_shape=(), dtype="float32"):
        self._kind = kind
        self._extra_shape = extra_shape
        self._dtype = np.dtype(dtype)
        self._data = None
        self._shape = None

    def set_shape_partition(self, partition):
        self._shape = self._shape_for_kind(self._kind, partition.shape)

    def set_shape_ds(self, dataset):
        self._shape = self._shape_for_kind(self._kind, dataset.shape)

    def _shape_for_kind(self, kind, orig_shape):
        if self._kind == "nav":
            return tuple(orig_shape.nav) + self._extra_shape
        elif self._kind == "sig":
            return tuple(orig_shape.sig) + self._extra_shape
        else:
            raise ValueError("unknown kind: %s" % kind)

    @property
    def data(self):
        return self._data

    def allocate(self):
        self._data = np.zeros(self._shape, dtype=self._dtype)

    def get_view_for_partition(self, partition):
        if self._kind == "nav":
            return self._data[partition.slice.get(nav_only=True)]
        elif self._kind == "sig":
            return self._data[partition.slice.get(sig_only=True)]

    def get_view_for_frame(self, partition, tile, frame_idx):
        if self._kind == "sig":
            return self._data[partition.slice.get(sig_only=True)]
        elif self._kind == "nav":
            ref_slice = partition.slice
            tile_slice = tile.tile_slice.shift(ref_slice)
            start_of_tile = np.ravel_multi_index(
                tile_slice.origin[:-tile_slice.shape.sig.dims],
                tuple(partition.shape.nav),
            )
            result_idx = np.unravel_index(start_of_tile + frame_idx,
                                          partition.shape.nav)
            # shape: (1,) + self._extra_shape
            if len(self._extra_shape) > 0:
                return self._data[result_idx]
            else:
                return self._data[result_idx + (np.newaxis,)]


def map_partition(partition, make_result_buffers, init_fn, frame_fn):
    result_buffers = make_result_buffers()
    for buf in result_buffers.values():
        buf.set_shape_partition(partition)
        buf.allocate()
    kwargs = init_fn(partition)
    kwargs.update(result_buffers)
    for tile in partition.get_tiles():
        data = tile.flat_nav
        for frame_idx, frame in enumerate(data):
            buffer_views = {}
            for k, buf in result_buffers.items():
                if buf._kind == "nav":
                    # import pdb
                    # pdb.set_trace()
                    pass
                buffer_views[k] = buf.get_view_for_frame(
                    partition=partition,
                    tile=tile,
                    frame_idx=frame_idx
                )
            kwargs.update(buffer_views)
            frame_fn(frame=frame, **kwargs)
    return result_buffers, partition


def map_partitions(dataset, executor, merge, make_result_buffers, init_fn, frame_fn):
    result_buffers = make_result_buffers()
    for buf in result_buffers.values():
        buf.set_shape_ds(dataset)
        buf.allocate()
    fn = functools.partial(
        map_partition,
        make_result_buffers=make_result_buffers,
        init_fn=init_fn,
        frame_fn=frame_fn
    )
    for partition_result_buffers, partition in executor.map_partitions(dataset=dataset, fn=fn):
        buffer_views = {}
        for k, buf in result_buffers.items():
            buffer_views[k] = buf.get_view_for_partition(partition=partition)
        merge(partition_result_buffers, **buffer_views)
    return result_buffers
