import pytest
from collections.abc import Sequence
import numpy as np
import scipy.sparse as sp
import sparse
from sparseconverter import (
    CUPY, CUPY_BACKENDS, NUMPY, SPARSE_BACKENDS, SPARSE_COO, get_device_class
)

from utils import _naive_mask_apply, _mk_random, set_device_class

from libertem.masks import circular
from libertem.common.sparse import to_dense, to_sparse, is_sparse
from libertem.common import Shape, Slice
from libertem.io.dataset.memory import MemoryDataSet
from libertem.udf.masks import ApplyMasksUDF
from libertem.udf import UDF, UDFMeta


def _run_mask_test_program(lt_ctx, dataset, mask, expected, do_sparse=True):
    dtype = UDF.USE_NATIVE_DTYPE

    analysis_default = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask], dtype=dtype
    )
    if do_sparse:
        analysis_sparse = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=[lambda: to_sparse(mask)], use_sparse=True,
            dtype=dtype
        )
    analysis_dense = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: to_dense(mask)], use_sparse=False,
        dtype=dtype
    )
    results_default = lt_ctx.run(analysis_default)
    if do_sparse:
        results_sparse = lt_ctx.run(analysis_sparse)
    results_dense = lt_ctx.run(analysis_dense)

    analysis_list_of_arr = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=lambda: [mask, mask], dtype=dtype
    )
    lt_ctx.run(analysis_list_of_arr)

    assert np.allclose(
        results_default.mask_0.raw_data,
        expected
    )
    if do_sparse:
        assert np.allclose(
            results_sparse.mask_0.raw_data,
            expected
        )
    assert np.allclose(
        results_dense.mask_0.raw_data,
        expected
    )


@pytest.mark.xfail
@pytest.mark.slow
def test_weird_partition_shapes_1_slow(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16), partition_shape=(16, 16, 2, 2))

    _run_mask_test_program(lt_ctx, dataset, mask, expected)

    p = next(dataset.get_partitions())
    t = next(p.get_tiles())
    assert tuple(t.tile_slice.shape) == (1, 1, 2, 2)


@pytest.mark.xfail
def test_weird_partition_shapes_1_fast(lt_ctx):
    # XXX MemoryDataSet is now using Partition3D and so on, so we can't create
    # partitions with weird shapes so easily anymore (in this case, partitioned in
    # the signal dimensions). maybe fix this with a custom DataSet impl that simulates this?
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(8, 16, 16), partition_shape=(16, 16, 8, 8))

    _run_mask_test_program(lt_ctx, dataset, mask, expected)

    p = next(dataset.get_partitions())
    t = next(p.get_tiles())
    assert tuple(t.tile_slice.shape) == (1, 8, 8, 8)


def test_normal_partition_shape(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(1, 16, 16), num_partitions=2)

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(ApplyMasksUDF(lambda x: None).get_backends())
)
@pytest.mark.parametrize(
    'dtype', ("<u2", "float32")
)
def test_single_frame_tiles(lt_ctx, backend, dtype):
    with set_device_class(get_device_class(backend)):
        if backend in SPARSE_BACKENDS:
            data = _mk_random(size=(16, 16, 16, 16), dtype=dtype, array_backend=SPARSE_COO)
        else:
            data = _mk_random(size=(16, 16, 16, 16), dtype=dtype, array_backend=NUMPY)

        mask = _mk_random(size=(16, 16))
        expected = _naive_mask_apply([mask], data)

        dataset = MemoryDataSet(
            data=data,
            tileshape=(1, 16, 16),
            num_partitions=2,
            array_backends=(backend, ) if backend is not None else None,
        )

        _run_mask_test_program(lt_ctx, dataset, mask, expected)


@pytest.mark.slow
def test_subframe_tiles_slow(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(1, 4, 4), num_partitions=2)

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


def test_subframe_tiles_fast(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask = _mk_random(size=(16, 16))
    expected = _naive_mask_apply([mask], data)

    dataset = MemoryDataSet(data=data, tileshape=(8, 4, 4), num_partitions=2)

    _run_mask_test_program(lt_ctx, dataset, mask, expected)


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(ApplyMasksUDF(lambda x: None).get_backends())
)
def test_mask_uint(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        if backend in SPARSE_BACKENDS:
            data = _mk_random(size=(16, 16, 16, 16), dtype="<u2", array_backend=SPARSE_COO)
        else:
            data = _mk_random(size=(16, 16, 16, 16), dtype="<u2", array_backend=NUMPY)

        mask = _mk_random(size=(16, 16)).astype("uint16")
        expected = _naive_mask_apply([mask], data)

        dataset = MemoryDataSet(
            data=data,
            tileshape=(4 * 4, 4, 4),
            num_partitions=2,
            array_backends=(backend, ) if backend is not None else None,
        )
        # We skip the sparse test with integer masks on CuPy
        # since cuyx.scipy.sparse doesn't support it
        do_sparse = backend not in CUPY_BACKENDS
        _run_mask_test_program(lt_ctx, dataset, mask, expected, do_sparse=do_sparse)


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(ApplyMasksUDF(lambda x: None).get_backends())
)
def test_endian(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        data = np.random.choice(a=0xFFFF, size=(16, 16, 16, 16)).astype(">u2")
        mask = _mk_random(size=(16, 16))
        expected = _naive_mask_apply([mask], data)

        dataset = MemoryDataSet(
            data=data,
            tileshape=(4 * 4, 4, 4),
            num_partitions=2,
            array_backends=(backend, ) if backend is not None else None
        )

        _run_mask_test_program(lt_ctx, dataset, mask, expected)


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(ApplyMasksUDF(lambda x: None).get_backends())
)
def test_signed(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        data = np.random.choice(a=0xFFFF, size=(16, 16, 16, 16)).astype("<i4")
        mask = _mk_random(size=(16, 16))
        expected = _naive_mask_apply([mask], data)

        # NOTE: we allow casting from int32 to float32 here, and may lose some
        # precision in case of data with large dynamic range
        dataset = MemoryDataSet(
            data=data, tileshape=(4 * 4, 4, 4), num_partitions=2,
            check_cast=False,
            array_backends=(backend, ) if backend is not None else None,
        )

        _run_mask_test_program(lt_ctx, dataset, mask, expected)


@pytest.mark.parametrize(
    # More in-depth tests below
    'backend', (None, NUMPY, CUPY)
)
def test_multi_masks(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        mask0 = _mk_random(size=(16, 16))
        mask1 = sp.csr_matrix(_mk_random(size=(16, 16)))
        mask2 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))
        expected = _naive_mask_apply([mask0, mask1, mask2], data)

        dataset = MemoryDataSet(
            data=data, tileshape=(4 * 4, 4, 4), num_partitions=2,
            array_backends=(backend, ) if backend is not None else None,
        )
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=[lambda: mask0, lambda: mask1, lambda: mask2],
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )
        assert np.allclose(
            results.mask_2.raw_data,
            expected[2],
        )


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(ApplyMasksUDF(lambda x: None).get_backends())
)
def test_multi_mask_stack_dense(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = _mk_random(size=(2, 16, 16))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(
            data=data, tileshape=(4 * 4, 4, 4), num_partitions=2,
            array_backends=(backend, ) if backend is not None else None
        )
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, mask_count=2,
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(ApplyMasksUDF(lambda x: None).get_backends())
)
def test_multi_mask_stack_sparse(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = sparse.COO.from_numpy(_mk_random(size=(2, 16, 16)))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(
            data=data, tileshape=(4 * 4, 4, 4), num_partitions=2,
            array_backends=(backend, ) if backend is not None else None
        )
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, mask_count=2,
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(ApplyMasksUDF(lambda x: None).get_backends())
)
def test_multi_mask_stack_force_sparse(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = _mk_random(size=(2, 16, 16))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(
            data=data, tileshape=(4 * 4, 4, 4), num_partitions=2,
            array_backends=(backend, ) if backend is not None else None
        )
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, use_sparse=True, mask_count=2
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(ApplyMasksUDF(lambda x: None).get_backends())
)
@pytest.mark.with_numba  # coverage for rmatmul implementation
def test_multi_mask_stack_force_scipy_sparse(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = _mk_random(size=(2, 16, 16))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, use_sparse='scipy.sparse', mask_count=2
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(ApplyMasksUDF(lambda x: None).get_backends())
)
@pytest.mark.with_numba  # coverage for rmatmul implementation
def test_multi_mask_stack_force_scipy_sparse_csc(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = _mk_random(size=(2, 16, 16))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, use_sparse='scipy.sparse.csc', mask_count=2
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(ApplyMasksUDF(lambda x: None).get_backends())
)
def test_multi_mask_stack_force_sparse_pydata(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = _mk_random(size=(2, 16, 16))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(
            data=data, tileshape=(4 * 4, 4, 4), num_partitions=2,
            array_backends=(backend, ) if backend is not None else None
        )
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, use_sparse='sparse.pydata', mask_count=2
        )
        if backend in CUPY_BACKENDS:
            with pytest.raises(ValueError):
                results = lt_ctx.run(analysis)
        else:
            results = lt_ctx.run(analysis)
            assert np.allclose(
                results.mask_0.raw_data,
                expected[0],
            )
            assert np.allclose(
                results.mask_1.raw_data,
                expected[1],
            )


@pytest.mark.parametrize(
    'backend', (None, ) + tuple(ApplyMasksUDF(lambda x: None).get_backends())
)
def test_multi_mask_stack_force_sparse_pydata_GCXS(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = _mk_random(size=(2, 16, 16))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(
            data=data, tileshape=(4 * 4, 4, 4), num_partitions=2,
            array_backends=(backend, ) if backend is not None else None
        )
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, use_sparse='sparse.pydata.GCXS', mask_count=2
        )
        if backend in CUPY_BACKENDS:
            with pytest.raises(ValueError):
                results = lt_ctx.run(analysis)
        else:
            results = lt_ctx.run(analysis)
            assert np.allclose(
                results.mask_0.raw_data,
                expected[0],
            )
            assert np.allclose(
                results.mask_1.raw_data,
                expected[1],
            )


@pytest.mark.parametrize(
    'backend', ['numpy', 'cupy']
)
def test_multi_mask_stack_force_dense(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        masks = sparse.COO.from_numpy(_mk_random(size=(2, 16, 16)))
        expected = _naive_mask_apply(masks, data)

        dataset = MemoryDataSet(
            data=data, tileshape=(4 * 4, 4, 4), num_partitions=2,
            array_backends=(backend, ) if backend is not None else None
        )
        analysis = lt_ctx.create_mask_analysis(
            dataset=dataset, factories=lambda: masks, use_sparse=False, mask_count=2
        )
        results = lt_ctx.run(analysis)

        assert np.allclose(
            results.mask_0.raw_data,
            expected[0],
        )
        assert np.allclose(
            results.mask_1.raw_data,
            expected[1],
        )


def test_multi_mask_autodtype(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    masks = _mk_random(size=(2, 16, 16))
    expected = _naive_mask_apply(masks, data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=lambda: masks
    )
    results = lt_ctx.run(analysis)

    assert results.mask_0.raw_data.dtype == np.result_type(np.float32, data.dtype, masks.dtype)

    assert np.allclose(
        results.mask_0.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1.raw_data,
        expected[1],
    )


def test_multi_mask_autodtype_wide(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="int64")
    masks = _mk_random(size=(2, 16, 16))
    expected = _naive_mask_apply(masks, data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=lambda: masks
    )
    results = lt_ctx.run(analysis)

    assert results.mask_0.raw_data.dtype == np.result_type(np.float64, data.dtype, masks.dtype)

    assert np.allclose(
        results.mask_0.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1.raw_data,
        expected[1],
    )


def test_multi_mask_autodtype_complex(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="complex64")
    masks = _mk_random(size=(2, 16, 16))
    expected = _naive_mask_apply(masks, data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(dataset=dataset, factories=lambda: masks)
    results = lt_ctx.run(analysis)

    assert results.mask_0_complex.raw_data.dtype.kind == 'c'
    assert results.mask_0_complex.raw_data.dtype == np.complex64

    assert np.allclose(
        results.mask_0_complex.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1_complex.raw_data,
        expected[1],
    )


def test_multi_mask_autodtype_complex_wide(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16))
    masks = _mk_random(size=(2, 16, 16), dtype="complex128")
    expected = _naive_mask_apply(masks, data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=lambda: masks
    )
    results = lt_ctx.run(analysis)

    assert results.mask_0_complex.raw_data.dtype.kind == 'c'
    assert results.mask_0_complex.raw_data.dtype == np.complex128

    assert np.allclose(
        results.mask_0_complex.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1_complex.raw_data,
        expected[1],
    )


def test_multi_mask_force_dtype(lt_ctx):
    force_dtype = np.dtype(np.int32)
    data = _mk_random(size=(16, 16, 16, 16), dtype="int16")
    masks = _mk_random(size=(2, 16, 16), dtype="bool")
    expected = _naive_mask_apply(masks.astype(force_dtype), data.astype(force_dtype))

    dataset = MemoryDataSet(
        data=data,
        tileshape=(4 * 4, 4, 4),
        num_partitions=2,
        # disable other input dtypes to not upcast to floats
        # for cupyx.scipy.sparse
        array_backends=(NUMPY, ),
    )
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=lambda: masks, dtype=force_dtype
    )
    results = lt_ctx.run(analysis)

    assert results.mask_0.raw_data.dtype.kind == force_dtype.kind
    assert results.mask_0.raw_data.dtype == force_dtype

    assert np.allclose(
        results.mask_0.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1.raw_data,
        expected[1],
    )


def test_avoid_calculating_masks_on_client(hdf5_ds_1, local_cluster_ctx):
    mask = _mk_random(size=(16, 16))
    # We have to start a local cluster so that the masks are
    # computed in a different process
    analysis = local_cluster_ctx.create_mask_analysis(
        dataset=hdf5_ds_1, factories=[lambda: mask], mask_count=1, mask_dtype=np.float32
    )
    udf = analysis.get_udf()
    local_cluster_ctx.run_udf(dataset=hdf5_ds_1, udf=udf)

    assert udf.masks._computed_masks is None


def test_avoid_calculating_masks_on_client_udf(hdf5_ds_1, local_cluster_ctx):
    mask = _mk_random(size=(16, 16))
    # We have to use a real cluster instead of InlineJobExecutor so that the masks are
    # computed in a different process
    analysis = local_cluster_ctx.create_mask_analysis(
        dataset=hdf5_ds_1, factories=[lambda: mask], mask_count=1, mask_dtype=np.float32
    )
    udf = analysis.get_udf()
    local_cluster_ctx.run_udf(udf=udf, dataset=hdf5_ds_1)
    assert udf._mask_container is None


def test_override_mask_dtype(lt_ctx):
    mask_dtype = np.float32
    data = _mk_random(size=(16, 16, 16, 16), dtype=mask_dtype)
    masks = _mk_random(size=(2, 16, 16), dtype=np.float64)
    expected = _naive_mask_apply(masks.astype(mask_dtype), data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=lambda: masks, mask_dtype=mask_dtype, mask_count=len(masks),
    )
    results = lt_ctx.run(analysis)

    assert results.mask_0.raw_data.dtype == mask_dtype

    assert np.allclose(
        results.mask_0.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1.raw_data,
        expected[1],
    )


def test_mask_udf(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = _mk_random(size=(16, 16))
    mask1 = sp.csr_matrix(_mk_random(size=(16, 16)))
    mask2 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))
    # The ApplyMasksUDF returns data with shape ds.shape.nav + (mask_count, ),
    # different from ApplyMasksJob
    expected = np.moveaxis(_naive_mask_apply([mask0, mask1, mask2], data), (0, 1), (2, 0))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    udf = ApplyMasksUDF(
        mask_factories=[lambda: mask0, lambda: mask1, lambda: mask2]
    )
    results = lt_ctx.run_udf(udf=udf, dataset=dataset)

    assert np.allclose(results['intensity'].data, expected)


def test_numpy_is_sparse():
    mask = _mk_random(size=(16, 16))
    assert not is_sparse(mask)


def test_scipy_is_sparse():
    mask = sp.csr_matrix(_mk_random(size=(16, 16)))
    assert is_sparse(mask)


def test_sparse_is_sparse():
    mask = sparse.COO.from_numpy(_mk_random(size=(16, 16)))
    assert is_sparse(mask)


def test_sparse_dok_is_sparse():
    mask = sparse.DOK.from_numpy(_mk_random(size=(16, 16)))
    assert is_sparse(mask)


def test_all_sparse_analysis(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = sp.csr_matrix(_mk_random(size=(16, 16)))
    mask1 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))
    expected = _naive_mask_apply([mask0, mask1], data)

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1],
    )
    results = lt_ctx.run(analysis)

    assert np.allclose(
        results.mask_0.raw_data,
        expected[0],
    )
    assert np.allclose(
        results.mask_1.raw_data,
        expected[1],
    )


def _mask_from_analysis(dataset, analysis):
    slice_ = Slice(
        origin=(0, 0, 0),
        shape=Shape((1, 16, 16), sig_dims=2),
    )
    udf = analysis.get_udf()
    meta = UDFMeta(
        partition_slice=None,
        dataset_shape=dataset.shape,
        roi=None,
        dataset_dtype=dataset.dtype,
        input_dtype=dataset.dtype,
        corrections=None,
    )
    udf.set_meta(meta)
    return udf.masks.get(slice_, transpose=True)


def test_uses_sparse_all_default(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = sp.csr_matrix(_mk_random(size=(16, 16)))
    mask1 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1]
    )

    assert is_sparse(_mask_from_analysis(dataset, analysis))


def test_uses_sparse_mixed_default(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = sp.csr_matrix(_mk_random(size=(16, 16)))
    mask1 = _mk_random(size=(16, 16))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1]
    )

    assert not is_sparse(_mask_from_analysis(dataset, analysis))


def test_uses_sparse_true(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = _mk_random(size=(16, 16))
    mask1 = _mk_random(size=(16, 16))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1], use_sparse=True
    )

    assert is_sparse(_mask_from_analysis(dataset, analysis))


def test_uses_scipy_sparse(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = _mk_random(size=(16, 16))
    mask1 = _mk_random(size=(16, 16))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1], use_sparse='scipy.sparse'
    )

    assert sp.issparse(_mask_from_analysis(dataset, analysis))


def test_uses_sparse_pydata(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = _mk_random(size=(16, 16))
    mask1 = _mk_random(size=(16, 16))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1], use_sparse='sparse.pydata'
    )

    assert isinstance(_mask_from_analysis(dataset, analysis), sparse.SparseArray)


def test_uses_scipy_sparse_false(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = sp.csr_matrix(_mk_random(size=(16, 16)))
    mask1 = sp.csr_matrix(_mk_random(size=(16, 16)))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1], use_sparse=False
    )

    assert not is_sparse(_mask_from_analysis(dataset, analysis))


def test_uses_sparse_sparse_false(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
    mask0 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))
    mask1 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))

    dataset = MemoryDataSet(data=data, tileshape=(4 * 4, 4, 4), num_partitions=2)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0, lambda: mask1], use_sparse=False
    )

    assert not is_sparse(_mask_from_analysis(dataset, analysis))


def test_masks_timeseries_2d_frames(lt_ctx):
    data = _mk_random(size=(16 * 16, 16, 16), dtype="<u2")
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, 16, 16),
        num_partitions=2
    )
    mask0 = _mk_random(size=(16, 16))
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0],
    )
    results = lt_ctx.run(analysis)
    assert results.mask_0.raw_data.shape == (256,)


def test_masks_spectrum_linescan(lt_ctx):
    data = _mk_random(size=(16 * 16, 16 * 16), dtype="<u2")
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, 16 * 16),
        num_partitions=2,
        sig_dims=1,
    )
    mask0 = _mk_random(size=(16 * 16, ))
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0],
    )
    results = lt_ctx.run(analysis)
    assert results.mask_0.raw_data.shape == (16 * 16,)


def test_masks_spectrum(lt_ctx):
    data = _mk_random(size=(16, 16, 16 * 16), dtype="<u2")
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, 16 * 16),
        num_partitions=2,
        sig_dims=1,
    )
    mask0 = _mk_random(size=(16 * 16, ))
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0],
    )
    results = lt_ctx.run(analysis)
    assert results.mask_0.raw_data.shape == (16, 16)


@pytest.mark.slow
def test_masks_hyperspectral(lt_ctx):
    data = _mk_random(size=(16, 16, 16, 16, 16), dtype="<u2")
    dataset = MemoryDataSet(
        data=data,
        tileshape=(1, 16, 16, 16),
        num_partitions=2,
        sig_dims=3,
    )
    mask0 = _mk_random(size=(16, 16, 16))
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0],
    )
    results = lt_ctx.run(analysis)
    assert results.mask_0.raw_data.shape == (16, 16)


def test_masks_complex_ds(lt_ctx, ds_complex):
    mask0 = _mk_random(size=(16, 16))
    analysis = lt_ctx.create_mask_analysis(
        dataset=ds_complex, factories=[lambda: mask0],
    )
    results = lt_ctx.run(analysis)
    assert results.mask_0.raw_data.shape == (16, 16)


def test_masks_complex_mask(lt_ctx, ds_complex):
    mask0 = _mk_random(size=(16, 16), dtype="complex64")
    analysis = lt_ctx.create_mask_analysis(
        dataset=ds_complex, factories=[lambda: mask0],
    )
    expected = _naive_mask_apply([mask0], ds_complex.data)
    results = lt_ctx.run(analysis)
    assert results.mask_0_complex.raw_data.shape == (16, 16)
    assert np.allclose(
        results.mask_0_complex.raw_data,
        expected
    )

    # also execute _run_mask_test_program to check sparse implementation.
    # _run_mask_test_program checks mask_0 result, which is np.abs(mask_0_complex)
    _run_mask_test_program(lt_ctx, ds_complex, mask0, np.abs(expected))


def test_numerics_fail(lt_ctx):
    dtype = 'float32'
    # Highest expected detector resolution
    RESOLUTION = 4096
    # Highest expected detector dynamic range
    RANGE = 1e6
    # default value for all cells
    VAL = 1.1

    data = np.full((2, 1, RESOLUTION, RESOLUTION), VAL, dtype=np.float32)
    data[0, 0, 0, 0] += VAL * RANGE
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, RESOLUTION, RESOLUTION),
        num_partitions=1,
        sig_dims=2,
    )
    mask0 = np.ones((RESOLUTION, RESOLUTION), dtype=np.float64)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0], mask_count=1, mask_dtype=dtype,
    )

    results = lt_ctx.run(analysis)
    expected = np.array([[
        [VAL*RESOLUTION**2 + VAL*RANGE],
        [VAL*RESOLUTION**2]
    ]])
    naive = _naive_mask_apply([mask0], data)
    naive_32 = _naive_mask_apply([mask0.astype(dtype)], data)
    # The masks are float64, that means the calculation is performed with high resolution
    # and the naive result should be correct
    assert np.allclose(expected, naive)
    # We make sure LiberTEM calculated this with the lower-precision dtype we set
    assert np.allclose(results.mask_0.raw_data, expected[0]) == np.allclose(naive_32, expected)
    # Confirm that the numerical precision is actually insufficient.
    # If this succeeds, we have to rethink the premise of this test.
    assert not np.allclose(results.mask_0.raw_data, expected[0])


def test_numerics_succeed(lt_ctx):
    dtype = 'float64'
    # Highest expected detector resolution
    RESOLUTION = 4096
    # Highest expected detector dynamic range
    RANGE = 1e6
    # default value for all cells
    VAL = 1.1

    data = np.full((2, 1, RESOLUTION, RESOLUTION), VAL, dtype=np.float32)
    data[0, 0, 0, 0] += VAL * RANGE
    dataset = MemoryDataSet(
        data=data,
        tileshape=(2, RESOLUTION, RESOLUTION),
        num_partitions=1,
        sig_dims=2,
    )
    mask0 = np.ones((RESOLUTION, RESOLUTION), dtype=np.float32)
    analysis = lt_ctx.create_mask_analysis(
        dataset=dataset, factories=[lambda: mask0], mask_count=1, mask_dtype=dtype,
    )

    results = lt_ctx.run(analysis)
    expected = np.array([[
        [VAL*RESOLUTION**2 + VAL*RANGE],
        [VAL*RESOLUTION**2]
    ]])
    naive = _naive_mask_apply([mask0.astype(dtype)], data.astype(dtype))

    assert np.allclose(expected, naive)
    assert np.allclose(expected[0], results.mask_0.raw_data)


@pytest.mark.parametrize(
    'kwargs', (
        {},
        {'use_sparse': True},
        {'use_sparse': False},
        {'use_sparse': 'sparse.pydata'},
        {'use_torch': True},
    )
)
@pytest.mark.parametrize(
    'backend', (None, NUMPY, CUPY)
)
def test_shifted_masks_constant_shifts(lt_ctx, kwargs, backend):
    with set_device_class(get_device_class(backend)):
        data = _mk_random(size=(16, 16, 16, 16), dtype="<u2")
        mask0 = _mk_random(size=(16, 16))
        mask1 = sp.csr_matrix(_mk_random(size=(16, 16)))
        mask2 = sparse.COO.from_numpy(_mk_random(size=(16, 16)))
        # The ApplyMasksUDF returns data with shape ds.shape.nav + (mask_count, ),
        expected = np.moveaxis(_naive_mask_apply(
            [mask0[0:15, 2:16], mask1[0:15, 2:16], mask2[0:15, 2:16]],
            data[..., 1:16, 0:14]
        ), (0, 1), (2, 0))

        dataset = MemoryDataSet(data=data, num_partitions=2)
        udf = ApplyMasksUDF(
            mask_factories=[lambda: mask0, lambda: mask1, lambda: mask2],
            shifts=np.round((1.1, -1.7)),
            **kwargs
        )
        if backend == CUPY and kwargs.get('use_sparse', False) in (True, 'sparse.pydata'):
            with pytest.raises(ValueError):
                # Implement like this so we can test the implementation
                # once CUPY + sparse.pydata is actually supported
                results = lt_ctx.run_udf(udf=udf, dataset=dataset)
            pytest.xfail('CuPy + sparse.pydata not yet supported')
        else:
            results = lt_ctx.run_udf(udf=udf, dataset=dataset)

        assert np.allclose(results['intensity'].data, expected)


def test_shifted_masks_scipy_sparse_raises(lt_ctx):
    with pytest.raises(ValueError):
        ApplyMasksUDF(
            mask_factories=lambda: np.ones((5, 6)),
            shifts=(1.2, -3.1),
            use_sparse='scipy.sparse',
        )


def _make_lambda(mask):
    return lambda: mask


def naive_shifted_mask_apply(
    masks: Sequence[np.ndarray],
    data: np.ndarray,
    shifts: np.ndarray,
) -> np.ndarray:
    """
    Manually compute the intersection of masks with data when
    shifts are applied, and return the dot product between them
    as an array of shape (num_frames, num_masks)

    Inputs must all be array-like with shapes
        masks: np.array(num_masks, h, w) or [num_masks * np.array(h, w)]
        data: (num_frames, h, w)
        shifts: (2,) or (num_frames, 2) of (y/x shifts of the mask)
    Sparse input masks or data are densified
    If shifts has shape (2,) then the same shift is applied to all frames/masks
    A zero-overlap between mask and frame returns zero
    for the dot product for that pair
    """
    # Densify inputs just in case!
    data = to_dense(data)
    masks = np.asarray([to_dense(m) for m in masks])
    num_frames, h, w = data.shape
    num_masks, mh, mw = masks.shape
    assert h == mh
    assert w == mw

    # Ensure we have one shift pair per frame even if constant shift
    if shifts.shape == (2,):
        shifts = np.repeat(shifts[np.newaxis, ...], num_frames, axis=0)
    assert shifts.shape == (num_frames, 2)

    expected = np.zeros((num_frames, num_masks), dtype=np.float64)
    for frame_idx, (dy, dx) in enumerate(shifts.astype(int)):
        # A positive shift value moves the mask down/right
        # so we need a slice from (0, 0) to (h-sy, w-sx)
        # A negative shift value moves the mask up/left
        # so we need a slice from (abs(sy), abs(sy)) to (h, w)
        # The min(max(...)) etc ensure we never slice beyond the
        # edge of the frame/mask in any situation
        my0 = min(max(0, -dy), h)  # the negative survives max(0, ...) only when dy is negative
        mx0 = min(max(0, -dx), w)  # the negative survives max(0, ...) only when dx is negative
        my1 = max(min(h, h - dy), 0)
        mx1 = max(min(w, w - dx), 0)
        # The frame is sliced in exactly the opposite way
        dy0 = min(max(0, dy), h)
        dx0 = min(max(0, dx), w)
        dy1 = max(min(h, h + dy), 0)
        dx1 = max(min(w, w + dx), 0)
        for mask_idx, mask in enumerate(masks):
            mask_sub = mask[np.s_[my0: my1, mx0: mx1]]
            if mask_sub.size == 0:
                # zero intersection, neutral element 0. already in expected
                continue
            data_sub = data[frame_idx][np.s_[dy0: dy1, dx0: dx1]]
            expected[frame_idx, mask_idx] = (mask_sub * data_sub).sum(axis=(-1, -2))
    return expected


def test_naive_shifted_apply():
    shifts = np.asarray([1, -2]).astype(int)
    mask_s = np.s_[0:15, 2:16]
    data_s = np.s_[:, 1:16, 0:14]
    data = _mk_random(size=(2, 16, 16))
    mask = _mk_random(size=(16, 16))
    assert np.allclose(
        naive_shifted_mask_apply([mask], data, shifts).squeeze(),
        (data[data_s] * mask[mask_s][np.newaxis, ...]).sum(axis=(-1, -2)),
    )


@pytest.mark.parametrize(
    'kwargs', (
        {},  # equivalent to use_sparse=None
        {'use_sparse': True},
        {'use_sparse': False},
        {'use_sparse': 'sparse.pydata'},
        {'use_torch': True},  # with use_sparse=None
    )
)
@pytest.mark.parametrize(
    'backend', (None, ) + ApplyMasksUDF(
        mask_factories=lambda: np.ones((1, 1, 1)), shifts=(1, -1),
    ).get_backends()
)
@pytest.mark.parametrize(
    'mask_types', [
        (np.asarray,),
        (sp.csr_matrix,),
        (np.asarray, sparse.COO.from_numpy),
        (sparse.COO.from_numpy, sp.csc_matrix,),
    ]
)
def test_shifted_masks_aux_shifts(lt_ctx, kwargs, backend, mask_types):
    with set_device_class(get_device_class(backend)):
        # nonsquare frames to test ordering is correct
        num_frames, h, w = 2, 18, 12
        data = np.random.uniform(size=(num_frames, h, w)).astype(np.float32)
        if backend in SPARSE_BACKENDS:
            data = _mk_random((num_frames, h, w), array_backend=SPARSE_COO, sparse_density=0.1)
        else:
            data = _mk_random((num_frames, h, w), array_backend=NUMPY)
        shifts = np.random.uniform(
            low=-5,
            high=5,
            size=(num_frames, 2),
        )

        masks = []
        mask_factories = []
        for mask_type in mask_types:
            mask = mask_type(_mk_random(size=(h, w)))
            masks.append(mask)
            mask_factories.append(_make_lambda(mask))

        expected = naive_shifted_mask_apply(masks, data, shifts)
        dataset = MemoryDataSet(
            data=data,
            array_backends=(backend, ) if backend is not None else None
        )
        udf = ApplyMasksUDF(
            mask_factories=mask_factories,
            shifts=ApplyMasksUDF.aux_data(
                data=shifts.ravel(),
                kind='nav',
                extra_shape=(2, ),
                dtype=float,
            ),
            **kwargs,
        )

        use_sparse = kwargs.get('use_sparse', None)  # default value
        # if forcing sparse on CuPy, raise, else densify to process
        if backend in CUPY_BACKENDS and use_sparse in (True, 'sparse.pydata'):
            with pytest.raises(ValueError):
                # Implement like this so we can test the implementation
                # once CUPY + sparse.pydata is actually supported
                results = lt_ctx.run_udf(udf=udf, dataset=dataset)
            pytest.xfail('CuPy + sparse.pydata not yet supported')
        else:
            results = lt_ctx.run_udf(udf=udf, dataset=dataset)

        assert np.allclose(results['intensity'].data, expected)


def test_shifted_masks_zero_overlap(lt_ctx):
    num_frames, h, w = 2, 18, 12
    data = _mk_random(size=(num_frames, h, w))

    dataset = MemoryDataSet(data=data)
    udf = ApplyMasksUDF(
        mask_factories=[lambda: _mk_random(size=(h, w))],
        shifts=(-20, 15),
    )

    results = lt_ctx.run_udf(udf=udf, dataset=dataset)
    assert np.allclose(results['intensity'].data, 0.)


@pytest.mark.parametrize(
    'backend', (None, NUMPY, CUPY)
)
def test_shifted_masks_stacked(lt_ctx, backend):
    with set_device_class(get_device_class(backend)):
        # nonsquare frames to test ordering is correct
        num_frames, h, w = 2, 18, 12
        data = _mk_random(size=(num_frames, h, w))
        shifts = np.random.uniform(
            low=-5.,
            high=5.,
            size=(num_frames, 2),
        )

        masks = _mk_random(size=(3, h, w))

        expected = naive_shifted_mask_apply(masks, data, shifts)
        dataset = MemoryDataSet(data=data)
        udf = ApplyMasksUDF(
            mask_factories=lambda: masks,
            shifts=ApplyMasksUDF.aux_data(
                data=shifts.ravel(),
                kind='nav',
                extra_shape=(2, ),
                dtype=float,
            ),
        )

        results = lt_ctx.run_udf(udf=udf, dataset=dataset)
        assert np.allclose(results['intensity'].data, expected)


def test_shifted_masks_descan(lt_ctx):
    # simulate a descan error on a constant frame
    # check that unshifted ApplyMasksUDF has aliasing effects and that
    # applying the shift values gives the correct constant result
    h, w = (9, 9)
    frame = np.zeros((h, w), dtype=int)
    frame = circular(4, 4, w, h, 1)
    frame_sum = frame.sum()
    frames = []
    shifts = np.moveaxis(np.mgrid[-2:4:2, -2:4:2], 0, -1)
    for yroll, xroll in shifts.reshape(-1, 2):
        frames.append(np.roll(frame, (yroll, xroll), axis=(0, 1)))
    data = np.stack(frames, axis=0)
    ds = lt_ctx.load('memory', data=data)
    mask = circular(4, 4, w, h, 2)
    udf = ApplyMasksUDF(mask_factories=[lambda: mask])
    results = lt_ctx.run_udf(udf=udf, dataset=ds)
    assert not (results['intensity'].data == frame_sum).all()
    assert results['intensity'].data.reshape(3, 3)[1, 1] == frame_sum
    udf = ApplyMasksUDF(
        mask_factories=[lambda: mask],
        shifts=ApplyMasksUDF.aux_data(
            data=shifts.ravel(),
            kind='nav',
            extra_shape=(2, ),
            dtype=int,
        ),
    )
    results = lt_ctx.run_udf(udf=udf, dataset=ds)
    assert (results['intensity'].data == frame_sum).all()


def test_backend_unknown():
    with pytest.raises(ValueError):
        ApplyMasksUDF(
            mask_factories=[lambda: np.ones((5, 5))],
            backends=('BACKEND_DOES_NOT_EXIST',)
        )
