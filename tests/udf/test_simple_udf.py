import pickle

import numpy as np
import pytest

from libertem.udf.base import UDF, UDFRunner
from libertem.udf.base import UDFMeta
from libertem.io.dataset.memory import MemoryDataSet
from libertem.utils.devices import detect
from libertem.common.backend import set_use_cpu, set_use_cuda
from libertem.common.buffers import reshaped_view

from utils import _mk_random, ValidationUDF


def test_validation(lt_ctx):
    """
    Test that the ValidationUDF works as designed
    """
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    udf = ValidationUDF(reference=data.reshape((-1, 16, 16)))
    res = lt_ctx.run_udf(dataset=dataset, udf=udf)
    assert res['nav_shape'].data.shape == (16, 16)

    with pytest.raises(AssertionError):
        data2 = data.copy()
        data2[7, 9, 13, 11] += 0.1
        udf = ValidationUDF(reference=data2.reshape((-1, 16, 16)))
        res = lt_ctx.run_udf(dataset=dataset, udf=udf)

    with pytest.raises(AssertionError):
        def badcompare(a, b):
            return False

        udf = ValidationUDF(
            reference=data.reshape((-1, 16, 16)),
            validation_function=badcompare
        )
        res = lt_ctx.run_udf(dataset=dataset, udf=udf)


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


def test_no_default_merge(lt_ctx):
    """
    Test forgotten merge function if not :code:`kind='nav'`.
    """
    data = _mk_random(size=(16 * 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    class NodefaultUDF(UDF):
        def get_result_buffers(self):
            return {
                'pixelsum_nav': self.buffer(
                    kind="nav", dtype="float32"
                ),
                'pixelsum': self.buffer(
                    kind="sig", dtype="float32"
                )
            }

        def process_frame(self, frame):
            self.results.pixelsum[:] += frame

    with pytest.raises(NotImplementedError):
        nd = NodefaultUDF()
        lt_ctx.run_udf(dataset=dataset, udf=nd)


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
    assert res['test'].extra_shape == (2,)
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
        dataset_dtype="float32",
        input_dtype="float32",
        device_class="cpu",
    )
    pixelsum.set_backend("numpy")
    pixelsum.set_meta(meta)
    pixelsum.init_result_buffers()
    pixelsum.allocate_for_part(partition, None)
    pickle.loads(pickle.dumps(pixelsum))


class ExtraShapeWithZero(UDF):
    def get_result_buffers(self):
        return {
            'testnav': self.buffer(
                kind="nav", dtype="float32", extra_shape=(0,)
            ),
            'testsig': self.buffer(
                kind="sig", dtype="float32", extra_shape=(0,)
            ),
            'testsingle': self.buffer(
                kind="single", dtype="float32", extra_shape=(0,)
            )
        }

    def process_frame(self, frame):
        pass

    def merge(self, dest, src):
        pass


def test_extra_shape_with_zero(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    udf = ExtraShapeWithZero()
    res = lt_ctx.run_udf(dataset=dataset, udf=udf)

    assert res['testnav'].data.size == 0
    assert res['testnav'].data.shape == tuple(dataset.shape.nav) + (0,)
    assert res['testsig'].data.size == 0
    assert res['testsig'].data.shape == tuple(dataset.shape.sig) + (0,)
    assert res['testsingle'].data.size == 0
    assert res['testsingle'].data.shape == (0,)


@pytest.mark.parametrize(
    'preferred_dtype,data_dtype,expected_dtype',
    [
        (bool, np.uint16, None),
        (None, bool, np.float32),
        (None, np.complex64, np.complex64),
        (None, np.complex128, np.complex128),
        (np.int16, np.uint16, None),
        (UDF.USE_NATIVE_DTYPE, np.int32, np.int32)
    ]
)
def test_dtypes(lt_ctx, preferred_dtype, data_dtype, expected_dtype):
    class DebugDTypeUDF(UDF):
        def __init__(self, preferred_dtype):
            super().__init__(preferred_dtype=preferred_dtype)

        def get_preferred_input_dtype(self):
            if self.params.preferred_dtype is None:
                return super().get_preferred_input_dtype()
            else:
                return self.params.preferred_dtype

        def get_result_buffers(self):
            return {
                'dtype': self.buffer(
                    kind='nav', dtype='object'
                ),
                'input_dtype': self.buffer(
                    kind='single', dtype=self.meta.input_dtype
                ),
                'dataset_dtype': self.buffer(
                    kind='single', dtype=self.meta.dataset_dtype
                )
            }

        def process_frame(self, frame):
            self.results.dtype[:] = frame.dtype
            assert frame.dtype == self.meta.input_dtype
            assert self.results.input_dtype.dtype == self.meta.input_dtype
            assert self.results.dataset_dtype.dtype == self.meta.dataset_dtype

        def merge(self, dest, src):
            dest['dtype'][:] = src['dtype'][:]

    if expected_dtype is None:
        expected_dtype = np.result_type(preferred_dtype, data_dtype)
    data = np.zeros(shape=(1, 1), dtype=data_dtype)
    dataset = MemoryDataSet(data=data, sig_dims=1)

    udf = DebugDTypeUDF(preferred_dtype=preferred_dtype)

    res = lt_ctx.run_udf(udf=udf, dataset=dataset)

    print(res['dtype'].data[0])
    assert res['dtype'].data[0] == expected_dtype
    assert udf.meta.dataset_dtype == data_dtype
    assert udf.meta.input_dtype == expected_dtype
    assert res['input_dtype'].data.dtype == expected_dtype
    assert res['dataset_dtype'].data.dtype == data_dtype


def test_with_progress_bar(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="float32")
    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16),
                            num_partitions=2, sig_dims=2)

    pixelsum = PixelsumUDF()
    res = lt_ctx.run_udf(dataset=dataset, udf=pixelsum, progress=True)
    # TODO: maybe assert that some output happened on stderr?


class ReshapedViewUDF(UDF):
    def get_result_buffers(self):
        return {
            "sigbuf": self.buffer(kind="sig", dtype=np.int, where="device")
        }

    def process_tile(self, tile):
        flat_tile = reshaped_view(tile, (tile.shape[0], -1))
        flat_buf = reshaped_view(self.results.sigbuf, (-1, ))
        flat_buf[:] = 1

    def merge(self, dest, src):
        dest["sigbuf"][:] = src["sigbuf"][:]

    def get_backends(self):
        return ('numpy', 'cupy')


@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
def test_noncontiguous_tiles(lt_ctx, backend):
    if backend == 'cupy':
        d = detect()
        cudas = detect()['cudas']
        if not d['cudas'] or not d['has_cupy']:
            pytest.skip("No CUDA device or no CuPy, skipping CuPy test")

    data = _mk_random(size=(30, 3, 7), dtype="float32")
    dataset = MemoryDataSet(
        data=data, tileshape=(3, 2, 2),
        num_partitions=2, sig_dims=2
    )
    try:
        if backend == 'cupy':
            set_use_cuda(cudas[0])
        udf = ReshapedViewUDF()
        res = lt_ctx.run_udf(udf=udf, dataset=dataset)
        partition = next(dataset.get_partitions())
        p_udf = udf.copy_for_partition(partition=partition, roi=None)
        # Enabling debug=True checks for disjoint cache keys
        part_res = UDFRunner([p_udf], debug=True).run_for_partition(
            partition=partition,
            roi=None,
            corrections=None,
        )

    finally:
        set_use_cpu(0)

    assert np.all(res["sigbuf"].data == 1)
