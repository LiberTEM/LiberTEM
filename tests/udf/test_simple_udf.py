import numpy as np
import pytest

from libertem.common.buffers import BufferWrapper

from utils import MemoryDataSet, _mk_random


def test_sum_frames(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 1, 16, 16),
                            partition_shape=(4, 4, 16, 16), sig_dims=2)

    def my_buffers():
        return {
            'pixelsum': BufferWrapper(
                kind="nav", dtype="float32"
            )
        }

    def my_frame_fn(frame, pixelsum):
        pixelsum[:] = np.sum(frame)

    res = lt_ctx.run_udf(
        dataset=dataset,
        fn=my_frame_fn,
        make_buffers=my_buffers,
    )
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(2, 3)))


def test_3d_ds(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            partition_shape=(4, 16, 16), sig_dims=2)

    def my_buffers():
        return {
            'pixelsum': BufferWrapper(
                kind="nav", dtype="float32"
            )
        }

    def my_frame_fn(frame, pixelsum):
        pixelsum[:] = np.sum(frame)

    res = lt_ctx.run_udf(
        dataset=dataset,
        fn=my_frame_fn,
        make_buffers=my_buffers,
    )
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(1, 2)))


def test_kind_single(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 2, 16, 16),
                            partition_shape=(4, 4, 16, 16), sig_dims=2)

    def counter_buffers():
        return {
            'counter': BufferWrapper(
                kind="single", dtype="uint32"
            )
        }

    def count_frames(frame, counter):
        counter += 1

    def merge_counters(dest, src):
        dest['counter'][:] += src['counter']

    res = lt_ctx.run_udf(
        dataset=dataset,
        fn=count_frames,
        make_buffers=counter_buffers,
        merge=merge_counters,
    )
    assert 'counter' in res
    assert res['counter'].data.shape == (1,)
    assert res['counter'].data == 16 * 16


def test_bad_merge(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            partition_shape=(4, 16, 16), sig_dims=2)

    def my_buffers():
        return {
            'pixelsum': BufferWrapper(
                kind="nav", dtype="float32"
            )
        }

    def my_frame_fn(frame, pixelsum):
        pixelsum[:] = np.sum(frame)

    def bad_merge(dest, src):
        # bad, because it just sets a key in dest, it doesn't copy over the data to dest
        dest['pixelsum'] = src['pixelsum']

    with pytest.raises(TypeError):
        lt_ctx.run_udf(
            dataset=dataset,
            fn=my_frame_fn,
            merge=bad_merge,
            make_buffers=my_buffers,
        )
