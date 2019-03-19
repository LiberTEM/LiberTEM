import functools
import numpy as np


def check_cast(fromvar, tovar):
    if not np.can_cast(fromvar.dtype, tovar.dtype, casting='safe'):
        # FIXME exception or warning?
        raise TypeError("Unsafe automatic casting from %s to %s" % (fromvar.dtype, tovar.dtype))


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
                buffer_views[k] = buf.get_view_for_frame(
                    partition=partition,
                    tile=tile,
                    frame_idx=frame_idx
                )
            kwargs.update(buffer_views)
            frame_fn(frame=frame, **kwargs)
    return result_buffers, partition


def merge_assign(dest, src):
    for k in dest:
        check_cast(dest[k], src[k])
        dest[k][:] = src[k]


def map_frames(ctx, dataset, make_result_buffers, init_fn, frame_fn, merge_fn=merge_assign):
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
    for partition_result_buffers, partition in ctx.executor.map_partitions(dataset=dataset, fn=fn):
        buffer_views = {}
        for k, buf in result_buffers.items():
            buffer_views[k] = buf.get_view_for_partition(partition=partition)
        buffers = {k: b.data
                   for k, b in partition_result_buffers.items()}
        merge_fn(dest=buffer_views, src=buffers)
    return result_buffers
