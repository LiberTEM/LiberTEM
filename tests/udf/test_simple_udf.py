import pickle

import numpy as np
import pytest

from libertem.udf import UDF
from libertem.udf.base import UDFMeta
from libertem.io.dataset.memory import MemoryDataSet

from utils import _mk_random


class PixelsumUDF(UDF):
    def get_result_buffers(self):
        return {
            'pixelsum': self.buffer(
                kind="nav", dtype="float32"
            )
        }

    def process_frame(self, frame):
        assert frame.shape == (16, 16)
        assert self.results.pixelsum.shape == (1,)
        self.results.pixelsum[:] = np.sum(frame)


def test_sum_frames(lt_ctx):
    """
    Test sum over the pixels for 2-dimensional dataset

    Parameters
    ----------
    lt_ctx
        Context class for loading dataset and creating jobs on them

    """
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    pixelsum = PixelsumUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum)
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(2, 3)))


def test_3d_ds(lt_ctx):
    """
    Test sum over the pixels for 3-dimensional dataset

    Parameters
    ----------
    lt_ctx
        Context class for loading dataset and creating jobs on them
    """
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    pixelsum = PixelsumUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum)
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    assert np.allclose(res['pixelsum'].data, np.sum(data, axis=(1, 2)))


def test_kind_single(lt_ctx):
    """
    Test buffer type kind='single'

    Parameters
    ----------
    lt_ctx
        Context class for loading dataset and creating jobs on them
    """
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(2, 16, 16),
                            num_partitions=2, sig_dims=2)

    class CounterUDF(UDF):
        def get_result_buffers(self):
            return {
                'counter': self.buffer(
                    kind="single", dtype="uint32"
                ),
                'sum_frame': self.buffer(
                    kind="single", extra_shape=(16,), dtype="float32"
                )
            }

        def process_frame(self, frame):
            self.results.counter[:] += 1
            self.results.sum_frame[:] += np.sum(frame, axis=1)

        def merge(self, dest, src):
            dest['counter'][:] += src['counter']
            dest['sum_frame'][:] += src['sum_frame']

    counter = CounterUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=counter)
    assert 'counter' in res
    assert 'sum_frame' in res
    assert res['counter'].data.shape == (1,)
    assert res['counter'].data == 16 * 16
    assert res['sum_frame'].data.shape == (16,)
    assert np.allclose(res['sum_frame'].data, np.sum(data, axis=(0, 1, 3)))


def test_bad_merge(lt_ctx):
    """
    Test bad example of updating buffer
    """
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    class BadmergeUDF(UDF):
        def get_result_buffers(self):
            return {
                'pixelsum': self.buffer(
                    kind="nav", dtype="float32"
                )
            }

        def process_frame(self, frame):
            self.results.pixelsum[:] = np.sum(frame)

        def merge(self, dest, src):
            # bad, because it just sets a key in dest, it doesn't copy over the data to dest
            dest['pixelsum'] = src['pixelsum']

    with pytest.raises(TypeError):
        bm = BadmergeUDF()
        lt_ctx.run_udf(dataset=dataset, udf=bm)


def test_extra_dimension_shape(lt_ctx):
    """
    Test sum over the pixels for 2-dimensional dataset

    Parameters
    ----------
    lt_ctx
        Context class for loading dataset and creating jobs on them

    """
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    class ExtraShapeUDF(UDF):
        def get_result_buffers(self):
            return {
                'test': self.buffer(
                    kind="nav", extra_shape=(2,), dtype="float32"
                ),
                'test2': self.buffer(
                    kind="sig", extra_shape=(2,), dtype="float32"
                ),
                'test3': self.buffer(
                    kind="single", extra_shape=(2,), dtype="float32"
                )
            }

        def process_frame(self, frame):
            self.results.test[:] = (1, 2)
            self.results.test2[:] += np.ones(tuple(self.meta.dataset_shape.sig) + (2, ))
            self.results.test3[:] += (1, 2)

        def merge(self, dest, src):
            dest['test'][:] = src['test'][:]
            dest['test2'][:] += src['test2'][:]
            dest['test3'][:] += src['test3'][:]

    extra = ExtraShapeUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=extra)

    navcount = np.prod(tuple(dataset.shape.nav))

    print(data.shape, res['test'].data.shape)
    assert res['test'].data.shape == tuple(dataset.shape.nav) + (2,)
    assert res['test2'].data.shape == tuple(dataset.shape.sig) + (2,)
    assert res['test3'].data.shape == (2,)
    assert np.allclose(res['test'].data, (1, 2))
    assert np.allclose(res['test2'].data, navcount)
    assert np.allclose(res['test3'].data, (navcount, 2*navcount))


def test_roi_1(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(3, 16, 16),
                            num_partitions=4, sig_dims=2)
    mask = np.random.choice([True, False], size=(16, 16))

    pixelsum = PixelsumUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum, roi=mask)
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    expected = np.sum(data[mask, ...], axis=(-1, -2))
    assert np.allclose(res['pixelsum'].raw_data, expected)


def test_roi_all_zeros(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(3, 16, 16),
                            num_partitions=16, sig_dims=2)
    mask = np.zeros(data.shape[:2], dtype=bool)

    pixelsum = PixelsumUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum, roi=mask)
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    expected = np.sum(data[mask, ...], axis=(-1, -2))
    assert np.allclose(res['pixelsum'].raw_data, expected)


def test_roi_all_ones(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(3, 16, 16),
                            num_partitions=16, sig_dims=2)
    mask = np.ones(data.shape[:2], dtype=bool)

    pixelsum = PixelsumUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum, roi=mask)
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    expected = np.sum(data[mask, ...], axis=(-1, -2))
    assert np.allclose(res['pixelsum'].raw_data, expected)


def test_roi_some_zeros(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(3, 16, 16),
                            num_partitions=16, sig_dims=2)
    mask = np.zeros(data.shape[:2], dtype=bool)
    mask[0] = True

    pixelsum = PixelsumUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum, roi=mask)
    assert 'pixelsum' in res
    print(data.shape, res['pixelsum'].data.shape)
    expected = np.sum(data[mask, ...], axis=(-1, -2))
    assert np.allclose(res['pixelsum'].raw_data, expected)


def test_roi_extra_dimension_shape(lt_ctx):
    """
    Test sum over the pixels for 2-dimensional dataset

    Parameters
    ----------
    lt_ctx
        Context class for loading dataset and creating jobs on them

    """
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    class ExtraShapeUDF(UDF):
        def get_result_buffers(self):
            return {
                'test': self.buffer(
                    kind="nav", extra_shape=(2,), dtype="float32"
                ),
                'test2': self.buffer(
                    kind="sig", extra_shape=(2,), dtype="float32"
                ),
                'test3': self.buffer(
                    kind="single", extra_shape=(2,), dtype="float32"
                )
            }

        def process_frame(self, frame):
            self.results.test[:] = (1, 2)
            self.results.test2[:] += np.ones(tuple(self.meta.dataset_shape.sig) + (2, ))
            self.results.test3[:] += (1, 2)

        def merge(self, dest, src):
            dest['test'][:] = src['test'][:]
            dest['test2'][:] += src['test2'][:]
            dest['test3'][:] += src['test3'][:]

    extra = ExtraShapeUDF()
    roi = _mk_random(size=dataset.shape.nav, dtype=np.bool)
    res = lt_ctx.run_udf(dataset=dataset, udf=extra, roi=roi)

    navcount = np.count_nonzero(roi)
    print(navcount)

    print(data.shape, res['test'].data.shape)
    assert res['test'].data.shape == tuple(dataset.shape.nav) + (2,)
    assert res['test2'].data.shape == tuple(dataset.shape.sig) + (2,)
    assert res['test3'].data.shape == (2,)
    assert np.allclose(res['test'].raw_data, (1, 2))
    assert np.allclose(res['test2'].raw_data, navcount)
    assert np.allclose(res['test3'].raw_data, (navcount, 2*navcount))


def test_udf_pickle(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(3, 16, 16),
                            num_partitions=16, sig_dims=2)

    partition = next(dataset.get_partitions())
    pixelsum = PixelsumUDF()
    meta = UDFMeta(
        partition_shape=partition.slice.shape,
        dataset_shape=dataset.shape,
        roi=None,
        dataset_dtype="float32"
    )
    pixelsum.set_meta(meta)
    pixelsum.init_result_buffers()
    pixelsum.allocate_for_part(partition, None)
    pickle.loads(pickle.dumps(pixelsum))


class InvalidBuffer(UDF):
    def get_result_buffers(self):
        return {
            'wrong': self.buffer(
                kind="nav", dtype="float32", extra_shape=(0,),
            )
        }

    def process_frame(self, frame):
        pass


def test_invalid_extra_shape(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    udf = InvalidBuffer()
    with pytest.raises(ValueError) as e:
        lt_ctx.run_udf(dataset=dataset, udf=udf)

    assert e.match("invalid extra_shape")
